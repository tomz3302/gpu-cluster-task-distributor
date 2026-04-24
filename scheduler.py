import asyncio
import httpx
import time
from pydantic import BaseModel

# --- Data Models (Based on Project Skeleton) ---
class RequestModel(BaseModel):
    id: int
    query: str

class Scheduler:
    def __init__(self, lb_url: str = "http://localhost:8080/process"):
        """
        The Scheduler initializes with the URL of the Load Balancer (HAProxy).
        """
        self.lb_url = lb_url
        # We use an AsyncClient for high-concurrency performance
        # Increased limits to handle 100+ RPS (default is 100)
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        self.client = httpx.AsyncClient(timeout=30.0, limits=limits)

    async def handle_request(self, request_data: RequestModel):
        """
        Receives a request and dispatches it to the Load Balancer.
        """
        print(f"[Master Scheduler] Dispatching request {request_data.id} to Load Balancer...")
        
        start_time = time.time()
        
        try:
            # The Scheduler forwards the task to HAProxy (The Load Balancer) 
            response = await self.client.post(
                self.lb_url, 
                json=request_data.dict()
            )
            
            if response.status_code == 200:
                result = response.json()
                total_latency = time.time() - start_time
                
                # Monitor performance
                print(f"[Master Scheduler] Request {request_data.id} completed by {result['worker_id']}")
                return {
                    **result,
                    "scheduler_overhead": round(total_latency - result['latency'], 4)
                }
            else:
                return {"error": f"Load Balancer returned status {response.status_code}"}
                
        except httpx.RequestError as exc:
            # Fault Tolerance: Detect if the Load Balancer or all nodes are down
            print(f"[Master Scheduler] Error: Could not connect to system. {exc}")
            return {"error": "System unavailable or timeout"}

    async def shutdown(self):
        """Cleanly close the network client."""
        await self.client.aclose()

# --- Example of how the Client Layer interacts with this---
async def main_demo():
    scheduler = Scheduler()
    
    # Simulate a single user request
    test_req = RequestModel(id=1, query="What is Distributed Computing?")
    response = await scheduler.handle_request(test_req)
    
    print("\n--- Final System Response ---")
    print(response)
    
    await scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main_demo())