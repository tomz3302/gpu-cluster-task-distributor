import os
import time
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ollama import Client


# -----------------------------
# Configuration
# -----------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "smollm:135m")

# Start with 1 for clean benchmarking.
# Later you can test 2, 4, etc.
MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "1"))

client = Client(host=OLLAMA_HOST)
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)

app = FastAPI(title="GPU Worker API", version="1.0")


# -----------------------------
# Metrics state
# -----------------------------

metrics_lock = asyncio.Lock()

metrics = {
    "active_requests": 0,
    "completed_requests": 0,
    "failed_requests": 0,
    "total_latency": 0.0,
    "total_queue_time": 0.0,
    "total_inference_time": 0.0,
}


# -----------------------------
# Request/Response Models
# -----------------------------

class GenerateRequest(BaseModel):
    id: int
    query: str
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    id: int
    model: str
    result: str
    queue_time: float
    inference_time: float
    total_latency: float


# -----------------------------
# Helper function for Ollama
# -----------------------------

def run_ollama_inference(query: str, max_tokens: int, temperature: float) -> str:
    """
    Blocking Ollama call.
    We run this inside asyncio.to_thread() so FastAPI does not block the event loop.
    """

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        options={
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    )

    # Works with recent ollama-python versions
    if hasattr(response, "message"):
        return response.message.content

    # Fallback for dict-style response
    return response["message"]["content"]


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "max_concurrent_inference": MAX_CONCURRENT_INFERENCE,
    }


@app.get("/metrics")
async def get_metrics():
    async with metrics_lock:
        completed = metrics["completed_requests"]

        if completed > 0:
            average_latency = metrics["total_latency"] / completed
            average_queue_time = metrics["total_queue_time"] / completed
            average_inference_time = metrics["total_inference_time"] / completed
        else:
            average_latency = 0.0
            average_queue_time = 0.0
            average_inference_time = 0.0

        return {
            "model": MODEL_NAME,
            "max_concurrent_inference": MAX_CONCURRENT_INFERENCE,
            "active_requests": metrics["active_requests"],
            "completed_requests": metrics["completed_requests"],
            "failed_requests": metrics["failed_requests"],
            "average_latency": average_latency,
            "average_queue_time": average_queue_time,
            "average_inference_time": average_inference_time,
        }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    request_start = time.perf_counter()

    async with metrics_lock:
        metrics["active_requests"] += 1

    try:
        # Queue starts when request arrives.
        queue_start = time.perf_counter()

        # Only allow a limited number of simultaneous LLM calls.
        async with inference_semaphore:
            queue_end = time.perf_counter()
            queue_time = queue_end - queue_start

            inference_start = time.perf_counter()

            result = await asyncio.to_thread(
                run_ollama_inference,
                request.query,
                request.max_tokens,
                request.temperature,
            )

            inference_end = time.perf_counter()
            inference_time = inference_end - inference_start

        total_latency = time.perf_counter() - request_start

        async with metrics_lock:
            metrics["completed_requests"] += 1
            metrics["total_latency"] += total_latency
            metrics["total_queue_time"] += queue_time
            metrics["total_inference_time"] += inference_time

        return GenerateResponse(
            id=request.id,
            model=MODEL_NAME,
            result=result,
            queue_time=queue_time,
            inference_time=inference_time,
            total_latency=total_latency,
        )

    except Exception as e:
        async with metrics_lock:
            metrics["failed_requests"] += 1

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        async with metrics_lock:
            metrics["active_requests"] -= 1
