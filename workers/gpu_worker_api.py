import os
import time
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ollama import Client

from rag.retriever import retrieve_context, build_rag_prompt


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "smollm:135m")
MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "4"))

WORKER_NAME = os.getenv("WORKER_NAME", "worker-local")

client = Client(host=OLLAMA_HOST)
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)

app = FastAPI(title="RAG GPU Worker API", version="1.0")


metrics_lock = asyncio.Lock()

# State for tracking idle time
worker_start_time = time.perf_counter()
active_inference_count = 0
busy_since = None

metrics = {
    "active_requests": 0,
    "completed_requests": 0,
    "failed_requests": 0,
    "total_latency": 0.0,
    "total_queue_time": 0.0,
    "total_retrieval_time": 0.0,
    "total_inference_time": 0.0,
    "total_busy_time": 0.0,
}


class GenerateRequest(BaseModel):
    id: int
    query: str
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_k: int = Field(default=3, ge=1, le=10)
    use_rag: bool = True


class GenerateResponse(BaseModel):
    id: int
    worker_name: str
    model: str
    result: str
    used_rag: bool
    sources: List[Dict[str, Any]]
    queue_time: float
    retrieval_time: float
    inference_time: float
    total_latency: float
    idle_time: float


def run_ollama_inference(prompt: str, max_tokens: int, temperature: float) -> str:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    )

    if hasattr(response, "message"):
        return response.message.content

    return response["message"]["content"]


def get_current_idle_time() -> float:
    now = time.perf_counter()
    total_time = now - worker_start_time
    
    # Calculate total busy time including current active ones
    current_busy_total = metrics["total_busy_time"]
    if active_inference_count > 0 and busy_since is not None:
        current_busy_total += now - busy_since
        
    return max(0.0, total_time - current_busy_total)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "worker_name": WORKER_NAME,
        "model": MODEL_NAME,
        "rag_enabled": True,
        "max_concurrent_inference": MAX_CONCURRENT_INFERENCE,
    }


@app.get("/metrics")
async def get_metrics():
    async with metrics_lock:
        completed = metrics["completed_requests"]

        if completed > 0:
            average_latency = metrics["total_latency"] / completed
            average_queue_time = metrics["total_queue_time"] / completed
            average_retrieval_time = metrics["total_retrieval_time"] / completed
            average_inference_time = metrics["total_inference_time"] / completed
        else:
            average_latency = 0.0
            average_queue_time = 0.0
            average_retrieval_time = 0.0
            average_inference_time = 0.0

        return {
            "model": MODEL_NAME,
            "max_concurrent_inference": MAX_CONCURRENT_INFERENCE,
            "active_requests": metrics["active_requests"],
            "completed_requests": metrics["completed_requests"],
            "failed_requests": metrics["failed_requests"],
            "average_latency": average_latency,
            "average_queue_time": average_queue_time,
            "average_retrieval_time": average_retrieval_time,
            "average_inference_time": average_inference_time,
            "total_idle_time": get_current_idle_time(),
        }


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
):
    global active_inference_count, busy_since
    request_start = time.perf_counter()

    async with metrics_lock:
        metrics["active_requests"] += 1

    try:
        queue_start = time.perf_counter()

        async with inference_semaphore:
            async with metrics_lock:
                if active_inference_count == 0:
                    busy_since = time.perf_counter()
                active_inference_count += 1

            try:
                queue_end = time.perf_counter()
                queue_time = queue_end - queue_start

                retrieval_time = 0.0
                sources = []

                if request.use_rag:
                    retrieval_result = await asyncio.to_thread(
                        retrieve_context,
                        request.query,
                        request.top_k,
                    )

                    retrieval_time = retrieval_result["retrieval_time"]
                    sources = retrieval_result["sources"]

                    prompt = build_rag_prompt(
                        request.query,
                        retrieval_result["context"],
                    )
                else:
                    prompt = request.query

                inference_start = time.perf_counter()

                result = await asyncio.to_thread(
                    run_ollama_inference,
                    prompt,
                    request.max_tokens,
                    request.temperature,
                )

                inference_end = time.perf_counter()
                inference_time = inference_end - inference_start
            finally:
                async with metrics_lock:
                    active_inference_count -= 1
                    if active_inference_count == 0:
                        if busy_since is not None:
                            metrics["total_busy_time"] += time.perf_counter() - busy_since
                            busy_since = None

        total_latency = time.perf_counter() - request_start

        async with metrics_lock:
            metrics["completed_requests"] += 1
            metrics["total_latency"] += total_latency
            metrics["total_queue_time"] += queue_time
            metrics["total_retrieval_time"] += retrieval_time
            metrics["total_inference_time"] += inference_time
            current_idle_time = get_current_idle_time()

        return GenerateResponse(
            id=request.id,
            worker_name=WORKER_NAME,
            model=MODEL_NAME,
            result=result,
            used_rag=request.use_rag,
            sources=sources,
            queue_time=queue_time,
            retrieval_time=retrieval_time,
            inference_time=inference_time,
            total_latency=total_latency,
            idle_time=current_idle_time,
        )

    except Exception as e:
        async with metrics_lock:
            metrics["failed_requests"] += 1

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        async with metrics_lock:
            metrics["active_requests"] -= 1


"""
EXPLAINER: The 'busy_since' Variable & Idle Tracking Logic

'busy_since' acts as a timestamp for when the worker transitions from an 
'Idle' state to a 'Busy' state. It follows a 'Last-In, First-Out' logic 
to ensure we don't double-count time when multiple requests are 
processing at once.

How it works:
1. THE START (0 -> 1): When a request clears the semaphore and finds that 
   'active_inference_count' is currently 0, it means the worker was idle. 
   We set 'busy_since' to the current time (time.perf_counter()).

2. THE OVERLAP: If a 2nd, 3rd, or 4th request starts while the first is 
   still running, the count increases, but 'busy_since' remains 
   untouched. We are already 'busy'.

3. THE END (1 -> 0): When a request finishes and decrements the count 
   to 0, it means the GPU is finally empty. At this point, we calculate:
   (Current Time - busy_since) = The duration of this busy "session".
   This duration is added to metrics["total_busy_time"], and 
   'busy_since' is reset to None.

4. IDLE CALCULATION: 
   Total Idle Time = (Total Worker Uptime) - (Total Busy Time).
   This allows the /metrics endpoint to report exactly how long the 
   worker has been sitting around doing nothing vs. actually crunching 
   tokens.
"""