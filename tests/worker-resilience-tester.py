"""
WORKER_RESILIENCE_TESTER.PY: Sustained Stress & User Simulation 

This script evolves the original 'load_test.py' from a simple burst tester into a 
Locust-style performance suite. While the original script is great for checking 
the absolute 'breaking point' of the GPU, this version measures how the cluster 
behaves under realistic, sustained human-like traffic.

KEY DIFFERENCES & ADVANTAGES:
-----------------------------
1. SUSTAINED VS. BURST: 
   Instead of sending a fixed pile of requests and stopping, this runs for a set 
   DURATION (e.g., 60s). This identifies long-term issues like thermal throttling, 
   memory leaks, or connection pool exhaustion.

2. USER REALISM (THINK TIME):
   Introduces 'THINK_TIME' between requests. This simulates real users reading 
   responses before asking the next question, testing the Load Balancer's ability 
   to manage many concurrent, long-lived connections.

3. TRUE CLUSTER UTILIZATION:
   Unlike the original which averaged 'odometer' snapshots, this tracks 'idle_time' 
   DELTAS per worker name. It accurately calculates how hard each individual GPU 
   worked during the test window, even as HAProxy shifts load between workers.

4. DYNAMIC SCALING AWARENESS:
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
import httpx
from typing import Dict, List

# --- Configuration ---
LB_URL = "http://127.0.0.1:8080/generate"
CONCURRENT_USERS = 20  # Total "Users" to simulate
TEST_DURATION = 60     # Run for 60 seconds
THINK_TIME = 1.0       # Seconds a user waits between requests
PROMPT = "What is the capital of France?"

class StatsRegistry:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.results = []
        self.worker_snapshots = {} # name -> [idle_times]

    def record(self, result: dict):
        self.results.append(result)
        if result["success"]:
            name = result["worker_name"]
            if name not in self.worker_snapshots:
                self.worker_snapshots[name] = []
            self.worker_snapshots[name].append(result["idle_time"])

    def get_summary(self, total_elapsed):
        successes = [r for r in self.results if r["success"]]
        latencies = [r["latency"] for r in successes]
        
        # Calculate Utilization per worker
        utilizations = []
        for name, snapshots in self.worker_snapshots.items():
            if len(snapshots) >= 2:
                idle_delta = snapshots[-1] - snapshots[0]
                # Busy Time / Total Test Time
                u = (total_elapsed - idle_delta) / total_elapsed
                utilizations.append(max(0, min(1, u)))

        return {
            "total_reqs": len(self.results),
            "success_rate": (len(successes) / len(self.results)) * 100 if self.results else 0,
            "avg_lat": statistics.mean(latencies) if latencies else 0,
            "p95_lat": sorted(latencies)[int(len(latencies)*0.95)-1] if latencies else 0,
            "rps": len(successes) / total_elapsed,
            "avg_util": statistics.mean(utilizations) if utilizations else 0,
            "workers": len(self.worker_snapshots)
        }

async def simulated_user(user_id: int, stats: StatsRegistry, client: httpx.AsyncClient, stop_event: asyncio.Event):
    """Simulates a single user looping through requests until the test ends."""
    while not stop_event.is_set():
        start = time.perf_counter()
        try:
            resp = await client.post(LB_URL, json={
                "id": user_id, 
                "query": PROMPT, 
                "use_rag": True
            }, timeout=30)
            
            latency = time.perf_counter() - start
            
            if resp.status_code == 200:
                data = resp.json()
                stats.record({
                    "success": True,
                    "latency": latency,
                    "worker_name": data["worker_name"],
                    "idle_time": data["idle_time"]
                })
            else:
                stats.record({"success": False, "latency": latency})
                
        except Exception:
            stats.record({"success": False, "latency": time.perf_counter() - start})

        # "Thinking Time" (Locust-style)
        await asyncio.sleep(THINK_TIME)

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

    print("\n" + "="*40)
    print(f"TEST COMPLETE ({total_time:.2f}s)")
    print("="*40)
    print(f"Requests:    {summary['total_reqs']}")
    print(f"RPS:         {summary['rps']:.2f}")
    print(f"Avg Latency: {summary['avg_lat']:.2f}s")
    print(f"P95 Latency: {summary['p95_lat']:.2f}s")
    print(f"Workers:     {summary['workers']}")
    print(f"Cluster Util: {summary['avg_util']*100:.1f}%")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())