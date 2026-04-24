import asyncio
import time
from scheduler import Scheduler, RequestModel

async def send_request(scheduler, user_id, results):
    """Sends a single request and logs metrics for the admin."""
    request = RequestModel(id=user_id, query=f"Sample LLM Query {user_id}")
    
    response = await scheduler.handle_request(request)
    
    if "error" in response:
        results["failures"] += 1
    else:
        results["successes"] += 1
        results["latencies"].append(response["latency"])
    
    # Progress indicator for every 100 requests
    if (results["successes"] + results["failures"]) % 100 == 0:
        print(f"--- Progress: {results['successes'] + results['failures']} requests sent ---")

async def run_load_test(rps=100, duration_seconds=10):
    """
    Orchestrates the high-concurrency simulation.
    """
    scheduler = Scheduler()
    results = {"successes": 0, "failures": 0, "latencies": []}
    tasks = []
    
    start_time = time.time()
    total_requests = rps * duration_seconds
    
    print(f"Starting Load Test: {rps} Requests Per Second for {duration_seconds} seconds...")
    
    for i in range(total_requests):
        # Create a non-blocking task for each request
        task = asyncio.create_task(send_request(scheduler, i, results))
        tasks.append(task)
        
        # Throttle the loop to exactly the desired RPS
        await asyncio.sleep(1 / rps)
    
    # Wait for all pending LLM inferences to complete
    await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    print_report(results, total_time)
    await scheduler.shutdown()

def print_report(results, total_time):
    """Summarizes performance metrics for evaluation."""
    avg_latency = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0
    actual_rps = (results["successes"] + results["failures"]) / total_time
    
    print("\n" + "="*40)
    print("FINAL PERFORMANCE REPORT")
    print("="*40)
    print(f"Total Requests: {results['successes'] + results['failures']}")
    print(f"Successful:     {results['successes']}")
    print(f"Failed:         {results['failures']}")
    print(f"Target RPS:     100")
    print(f"Actual RPS:     {actual_rps:.2f}")
    print(f"Total Time:     {total_time:.2f} seconds")
    print(f"Avg Latency:    {avg_latency:.3f} seconds")
    print("="*40)

if __name__ == "__main__":
    try:
        asyncio.run(run_load_test(rps=100, duration_seconds=10))
    except KeyboardInterrupt:
        print("\nLoad test stopped by user.")