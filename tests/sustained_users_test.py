"""
WORKER_RESILIENCE_TESTER.PY: Sustained Stress & User Simulation 

This script evolves the original 'load_test.py' from a simple burst tester into a 
 performance suite. While the original script is great for checking 
the absolute 'breaking point' of the GPU, this version measures how the cluster 
behaves under realistic, sustained human-like traffic.

KEY DIFFERENCES:
-----------------------------
1. SUSTAINED VS. BURST: 
   Instead of sending a fixed pile of requests and stopping, this runs for a set 
   DURATION (e.g., 60s).

2. USER REALISM (THINK TIME):
   Introduces 'THINK_TIME' between requests. This simulates real users reading 
   responses before asking the next question, testing the Load Balancer's ability 
   to manage many concurrent, long-lived connections.

3. DYNAMIC SCALING AWARENESS:
   Automatically detects and groups metrics by 'worker_name'. If workers are added 
   or removed during the test, the metrics reflect the average health of the 
   active cluster rather than getting skewed by historical data.

METRICS CAPTURED:
-----------------
- RPS (Requests Per Second): Sustained throughput over time.
- P95 Latency: The 95th percentile "bad day" experience for a user.
- Multi-Worker Utilization: Percentage of time the GPU cluster was crunching tokens.
- Success Rate: Percentage of requests that cleared both HAProxy and the Worker.
"""


import asyncio
import time
import statistics
import json
import random
import httpx
import collections
from typing import Dict, List

# --- Configuration ---
LB_URL = "http://127.0.0.1:8080/generate"
CONCURRENT_USERS = 1000  # Total "Users" to simulate (Reduced from 1000)
TEST_DURATION = 60      # Run for 60 seconds
THINK_TIME_MIN = 10.0    # Minimum seconds a user waits
THINK_TIME_MAX = 30.0   # Maximum seconds a user waits

# Load questions from JSON
QUESTIONS_FILE = "tests/questions.json"
try:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        QUESTION_POOL = json.load(f)
except Exception as e:
    print(f"Warning: Could not load {QUESTIONS_FILE}: {e}")
    QUESTION_POOL = ["What is the capital of France?"]

class StatsRegistry:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.results = []
        self.worker_stats = collections.defaultdict(lambda: {
            "latencies": [],
            "queue_times": [],
            "retrieval_times": [],
            "inference_times": [],
            "gpu_utils": []
        })

    def record(self, result: dict):
        self.results.append(result)
        if result["success"]:
            name = result["worker_name"]
            ws = self.worker_stats[name]
            ws["latencies"].append(result["latency"])
            ws["queue_times"].append(result["queue_time"])
            ws["retrieval_times"].append(result["retrieval_time"])
            ws["inference_times"].append(result["inference_time"])
            ws["gpu_utils"].append(result["gpu_util"])

    def get_summary(self, total_elapsed):
        successes = [r for r in self.results if r["success"]]
        failures  = [r for r in self.results if not r["success"]]
        latencies = [r["latency"] for r in successes]
        queue_times     = [r["queue_time"]     for r in successes]
        retrieval_times = [r["retrieval_time"] for r in successes]
        inference_times = [r["inference_time"] for r in successes]
        
        # Calculate Utilization and Metrics per worker
        worker_details = {}
        utilizations = []
        for name, stats in self.worker_stats.items():
            if stats["gpu_utils"]:
                # Use hardware reported average
                u = statistics.mean(stats["gpu_utils"]) / 100.0
                u = max(0.0, min(1.0, u))
                utilizations.append(u)
                
                worker_details[name] = {
                    "utilization": u,
                    "avg_lat": statistics.mean(stats["latencies"]),
                    "avg_ret": statistics.mean(stats["retrieval_times"]),
                    "avg_inf": statistics.mean(stats["inference_times"])
                }
            else:
                worker_details[name] = None

        return {
            "total_reqs": len(self.results),
            "success_count": len(successes),
            "failure_count": len(failures),
            "success_rate": (len(successes) / len(self.results)) * 100 if self.results else 0,
            "avg_lat": statistics.mean(latencies) if latencies else 0,
            "min_lat": min(latencies) if latencies else 0,
            "max_lat": max(latencies) if latencies else 0,
            "p95_lat": sorted(latencies)[int(len(latencies)*0.95)-1] if latencies else 0,
            "avg_queue": statistics.mean(queue_times) if queue_times else 0,
            "avg_retrieval": statistics.mean(retrieval_times) if retrieval_times else 0,
            "avg_inference": statistics.mean(inference_times) if inference_times else 0,
            "rps": len(successes) / total_elapsed,
            "avg_util": statistics.mean(utilizations) if utilizations else 0,
            "worker_details": worker_details,
            "workers": len(self.worker_stats),
            "example_error": failures[0]["error"] if failures else None
        }

async def simulated_user(user_id: int, stats: StatsRegistry, client: httpx.AsyncClient, stop_event: asyncio.Event):
    """Simulates a single user looping through requests until the test ends."""
    while not stop_event.is_set():
        prompt = random.choice(QUESTION_POOL)
        start = time.perf_counter()
        try:
            resp = await client.post(LB_URL, json={
                "id": user_id, 
                "query": prompt, 
                "use_rag": True
            }, timeout=180)
            
            latency = time.perf_counter() - start
            
            if resp.status_code == 200:
                data = resp.json()
                stats.record({
                    "success": True,
                    "latency": latency,
                    "worker_name": data.get("worker_name", "unknown"),
                    "gpu_util": data.get("gpu_utilization", 0.0),
                    "queue_time": data.get("queue_time", 0.0),
                    "retrieval_time": data.get("retrieval_time", 0.0),
                    "inference_time": data.get("inference_time", 0.0)
                })
            else:
                stats.record({
                    "success": False, 
                    "latency": latency,
                    "error": resp.text
                })
                
        except Exception as e:
            stats.record({
                "success": False, 
                "latency": time.perf_counter() - start,
                "error": str(e)
            })

        # "Thinking Time" (Locust-style)
        think_time = random.uniform(THINK_TIME_MIN, THINK_TIME_MAX)
        await asyncio.sleep(think_time)


async def main():
    stats = StatsRegistry()
    stop_event = asyncio.Event()
    
    print(f"#### Starting Test: {CONCURRENT_USERS} users for {TEST_DURATION}s ####")
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONCURRENT_USERS)) as client:
        # Spawn users
        tasks = [
            asyncio.create_task(simulated_user(i, stats, client, stop_event)) 
            for i in range(CONCURRENT_USERS)
        ]
        
        # Run for the specified duration
        await asyncio.sleep(TEST_DURATION)
        stop_event.set() # Tell users to stop
        
        # Wait for current requests to finish
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - stats.start_time
    summary = stats.get_summary(total_time)

    print("\n" + "="*70)
    print(f"RESILIENCE TEST COMPLETE ({total_time:.2f}s)")
    print("="*70)
    print(f"Successful requests:     {summary['success_count']}")
    print(f"Failed requests:         {summary['failure_count']}")
    print(f"Throughput:              {summary['rps']:.2f} req/s")
    print(f"Average latency:         {summary['avg_lat']:.2f} s")
    print(f"P95 latency:             {summary['p95_lat']:.2f} s")
    print(f"Min latency:             {summary['min_lat']:.2f} s")
    print(f"Max latency:             {summary['max_lat']:.2f} s")
    print(f"Average queue time:      {summary['avg_queue']:.2f} s")
    print(f"Average retrieval time:  {summary['avg_retrieval']:.2f} s")
    print(f"Average inference time:  {summary['avg_inference']:.2f} s")
    print(f"Cluster Utilization:     {summary['avg_util'] * 100:.2f}%")
    print(f"Workers detected:        {summary['workers']}")
    
    for worker, details in summary['worker_details'].items():
        if details is not None:
            print(f"  - {worker}:")
            print(f"      Utilization: {details['utilization'] * 100:.2f}%")
            print(f"      Avg Latency: {details['avg_lat']:.3f}s | Retr: {details['avg_ret']:.3f}s | Infer: {details['avg_inf']:.3f}s")
        else:
            print(f"  - {worker}: Insufficient data")

    if summary["example_error"]:
        print(f"Example error:           {summary['example_error']}")

    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())