import asyncio
import random
import time
import argparse
import os
from fastapi import FastAPI
from pydantic import BaseModel

# --- Placeholder for your future RAG logic ---
class RAGModule:
    async def query(self, user_query: str):
        # This is where your vector DB retrieval logic will go
        return f"Context found for '{user_query}' in knowledge base."

# --- Data Models ---
class LLMRequest(BaseModel):
    id: int
    query: str

class LLMResponse(BaseModel):
    id: int
    result: str
    latency: float
    worker_id: str

app = FastAPI()
rag = RAGModule()

# Limit concurrent "inferences" to force queuing
CONCURRENCY_LIMIT = 2
gpu_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
active_inferences = 0

import socket

# Argument parser to set the port when running from terminal
parser = argparse.ArgumentParser()
# Use hostname as the default ID so scaled containers have unique IDs
parser.add_argument("--id", type=str, default=os.getenv("WORKER_ID", socket.gethostname()), help="Unique ID for this worker")
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8001)), help="Port to run the worker on")
args, _ = parser.parse_known_args()

@app.post("/process", response_model=LLMResponse)
async def process_request(request: LLMRequest):
    global active_inferences
    start_time = time.time() 
    
    queue_start = time.time()
    async with gpu_semaphore:
        queue_time = time.time() - queue_start
        active_inferences += 1
        # Log to Docker console
        print(f"[{args.id}] Processing request {request.id}. Queue time: {queue_time:.2f}s. Active: {active_inferences}")
        
        # Constant delay for predictable testing (1 second)
        await asyncio.sleep(random.uniform(0.5, 2.5))
        
        active_inferences -= 1
    
    latency = time.time() - start_time
    return {
        "id": request.id,
        "result": f"Answer to {request.id}",
        "latency": round(latency, 3),
        "worker_id": args.id
    }

if __name__ == "__main__":
    import uvicorn
    # When running manually via python worker.py
    # Using string import for app to support reload=True
    uvicorn.run("worker:app", host="0.0.0.0", port=args.port, reload=True)
