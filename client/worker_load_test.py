import asyncio
import csv
import statistics
import time
import os
from pathlib import Path

import httpx


WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8000/generate")

PROMPT = "Explain what load balancing is in one short paragraph."
MAX_TOKENS = 64
TEMPERATURE = 0.2

RESULTS_FILE = Path("worker_benchmark_results.csv")


def percentile(values, percentile_value):
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int((percentile_value / 100) * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


async def send_request(client, request_id):
    start = time.perf_counter()

    try:
        response = await client.post(
            WORKER_URL,
            json={
                "id": request_id,
                "query": PROMPT,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": 3,
                "use_rag": True,
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
        total_latency = time.perf_counter() - start

        return {
            "success": False,
            "latency": total_latency,
            "queue_time": 0.0,
            "inference_time": 0.0,
            "error": str(e),
        }


async def run_test(concurrency, total_requests):
    print("=" * 70)
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

        tasks = [
            bounded_request(i)
            for i in range(total_requests)
        ]

        results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    latencies = [r["latency"] for r in successes]
    queue_times = [r["queue_time"] for r in successes]
    inference_times = [r["inference_time"] for r in successes]

    successful_requests = len(successes)
    failed_requests = len(failures)

    throughput = successful_requests / total_time if total_time > 0 else 0.0

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = percentile(latencies, 95)

    avg_queue_time = statistics.mean(queue_times) if queue_times else 0.0
    avg_inference_time = statistics.mean(inference_times) if inference_times else 0.0

    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

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
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "total_time": round(total_time, 4),
        "throughput_req_per_sec": round(throughput, 4),
        "avg_latency_sec": round(avg_latency, 4),
        "p95_latency_sec": round(p95_latency, 4),
        "min_latency_sec": round(min_latency, 4),
        "max_latency_sec": round(max_latency, 4),
        "avg_queue_time_sec": round(avg_queue_time, 4),
        "avg_inference_time_sec": round(avg_inference_time, 4),
    }


def save_results(rows):
    fieldnames = [
        "concurrency",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "total_time",
        "throughput_req_per_sec",
        "avg_latency_sec",
        "p95_latency_sec",
        "min_latency_sec",
        "max_latency_sec",
        "avg_queue_time_sec",
        "avg_inference_time_sec",
    ]

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved benchmark results to: {RESULTS_FILE}")


async def main():
    test_plan = [
        (50, 50),
        (100, 100),
        (200, 200),
        (300, 300),
    ]

    all_results = []

    for concurrency, total_requests in test_plan:
        result = await run_test(concurrency, total_requests)
        all_results.append(result)

        # Small pause so the worker/GPU can settle before the next test.
        await asyncio.sleep(5)

    save_results(all_results)


if __name__ == "__main__":
    asyncio.run(main())