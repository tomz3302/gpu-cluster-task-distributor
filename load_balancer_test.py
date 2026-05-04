import asyncio
import statistics
import time

import httpx

# ---------------------------------------------------------------
# Configuration — Point this to your HAProxy Load Balancer
# ---------------------------------------------------------------

# By default, HAProxy frontend is bound to *:8080
LB_BASE_URL = "http://127.0.0.1:8080"
LB_URL = f"{LB_BASE_URL}/generate"

PROMPT = "Explain what load balancing is in one short paragraph."
MAX_TOKENS = 64
TEMPERATURE = 0.2

REQUEST_HEADERS = {
    "ngrok-skip-browser-warning": "true",
}

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def percentile(values, percentile_value):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((percentile_value / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def print_separator():
    print("=" * 70)


# ---------------------------------------------------------------
# Single request
# ---------------------------------------------------------------

async def send_request(client: httpx.AsyncClient, request_id: int) -> dict:
    start = time.perf_counter()
    try:
        response = await client.post(
            LB_URL,
            headers=REQUEST_HEADERS,
            json={
                "id": request_id,
                "query": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            timeout=180,
        )
        total_latency = time.perf_counter() - start

        if response.status_code != 200:
            return {
                "success": False,
                "latency": total_latency,
                "queue_time": 0.0,
                "inference_time": 0.0,
                "error": response.text,
            }

        data = response.json()
        return {
            "success": True,
            "latency": total_latency,
            "queue_time": data.get("queue_time", 0.0),
            "inference_time": data.get("inference_time", 0.0),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "latency": time.perf_counter() - start,
            "queue_time": 0.0,
            "inference_time": 0.0,
            "error": str(e),
        }


# ---------------------------------------------------------------
# Run one concurrency level
# ---------------------------------------------------------------

async def run_test(concurrency: int, total_requests: int) -> dict:
    print_separator()
    print(f"Running test: concurrency={concurrency}, total_requests={total_requests}")

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    start_time = time.perf_counter()

    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(request_id):
            async with semaphore:
                return await send_request(client, request_id)

        results = await asyncio.gather(*[bounded_request(i) for i in range(total_requests)])

    total_time = time.perf_counter() - start_time

    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]

    latencies       = [r["latency"]        for r in successes]
    queue_times     = [r["queue_time"]     for r in successes]
    inference_times = [r["inference_time"] for r in successes]

    successful_requests = len(successes)
    failed_requests     = len(failures)
    throughput          = successful_requests / total_time if total_time > 0 else 0.0

    avg_latency        = statistics.mean(latencies)       if latencies       else 0.0
    p95_latency        = percentile(latencies, 95)
    min_latency        = min(latencies)                   if latencies       else 0.0
    max_latency        = max(latencies)                   if latencies       else 0.0
    avg_queue_time     = statistics.mean(queue_times)     if queue_times     else 0.0
    avg_inference_time = statistics.mean(inference_times) if inference_times else 0.0

    print(f"Successful requests:     {successful_requests}")
    print(f"Failed requests:         {failed_requests}")
    print(f"Total time:              {total_time:.2f} s")
    print(f"Throughput:              {throughput:.2f} req/s")
    print(f"Average latency:         {avg_latency:.2f} s")
    print(f"P95 latency:             {p95_latency:.2f} s")
    print(f"Min latency:             {min_latency:.2f} s")
    print(f"Max latency:             {max_latency:.2f} s")
    print(f"Average queue time:      {avg_queue_time:.2f} s")
    print(f"Average inference time:  {avg_inference_time:.2f} s")

    if failures:
        print(f"Example error:           {failures[0]['error']}")

    return {
        "concurrency":             concurrency,
        "total_requests":          total_requests,
        "successful_requests":     successful_requests,
        "failed_requests":         failed_requests,
        "total_time":              round(total_time,        4),
        "throughput_req_per_sec":  round(throughput,        4),
        "avg_latency_sec":         round(avg_latency,       4),
        "p95_latency_sec":         round(p95_latency,       4),
        "min_latency_sec":         round(min_latency,       4),
        "max_latency_sec":         round(max_latency,       4),
        "avg_queue_time_sec":      round(avg_queue_time,    4),
        "avg_inference_time_sec":  round(avg_inference_time,4),
    }


# ---------------------------------------------------------------
# Print final summary table
# ---------------------------------------------------------------

def print_summary(all_results: list[dict]):
    print_separator()
    print("BENCHMARK SUMMARY")
    print_separator()

    col_w = [8, 8, 8, 8, 10, 10, 10, 10, 12, 12]
    headers = [
        "Conc.", "Total", "OK", "Fail",
        "Time(s)", "Req/s", "Avg(s)", "P95(s)",
        "Queue(s)", "Infer(s)",
    ]
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(row_fmt.format(*headers))
    print("-" * 100)

    for r in all_results:
        print(row_fmt.format(
            r["concurrency"],
            r["total_requests"],
            r["successful_requests"],
            r["failed_requests"],
            r["total_time"],
            r["throughput_req_per_sec"],
            r["avg_latency_sec"],
            r["p95_latency_sec"],
            r["avg_queue_time_sec"],
            r["avg_inference_time_sec"],
        ))

    print_separator()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

async def main():
    # Verify the load balancer is reachable before starting
    print(f"Connecting to Load Balancer at: {LB_BASE_URL}")
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{LB_BASE_URL}/health", timeout=10)
            if health.status_code == 200:
                print(f"Load Balancer is reachable and returning health check responses.")
            else:
                print(f"Warning: LB returned status {health.status_code}")
    except Exception as e:
        print(f"ERROR: Could not reach Load Balancer — {e}")
        print("Make sure HAProxy is running and LB_BASE_URL matches the bound port.")
        return

    test_plan = [
        (1,    1),
        (2,    2),
        (4,    4),
        (8,    8),
        (16,   16),
        (32,   32),
        (64,   64),
        (128,  128),
        (256,256),
        (512,  512),
        (1000, 1000),
    ]

    all_results = []
    for concurrency, total_requests in test_plan:
        result = await run_test(concurrency, total_requests)
        all_results.append(result)
        await asyncio.sleep(5)   # let the GPUs settle

    print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
