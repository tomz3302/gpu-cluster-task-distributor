# Efficient Distributed LLM Inference Cluster
## High-Concurrency GPU Task Distribution with HAProxy & Docker

This project implements a distributed system capable of handling **1000+ concurrent user requests** involving Large Language Model (LLM) inference and Retrieval-Augmented Generation (RAG). 

The architecture uses **HAProxy** as a professional-grade load balancer to distribute tasks across a dynamic cluster of simulated GPU worker nodes.

---

## System Architecture
1.  **Client Layer (`main.py`)**: Simulates 100+ Requests Per Second (RPS) using asynchronous Python.
2.  **Scheduler (`scheduler.py`)**: Acts as the Master Node, dispatching requests to the Load Balancer.
3.  **Load Balancer (HAProxy)**: Handles Round-Robin distribution, health checks, and dynamic service discovery.
4.  **Worker Nodes (`worker.py`)**: Simulated GPU nodes running FastAPI. They use an `asyncio.Semaphore` to simulate hardware resource constraints (e.g., VRAM/Compute) and queuing.

---

## Prerequisites
- **Docker Desktop** installed and running.
- **Python 3.9+** installed locally.
- Access to a terminal (PowerShell, Bash, or CMD).

---

## Initialization & Setup

### 1. Pull Required Images
Ensure you have the official HAProxy image available:
```powershell
docker pull haproxy:latest
```

### 2. Prepare the Project Directory
Navigate to your project folder:
```powershell
cd path/to/your/project/
```

### 3. Install Local Dependencies
The Client/Scheduler runs on your host machine. Install the required libraries:
```powershell
pip install fastapi uvicorn httpx pydantic
```

---

## Building and Running the Cluster

### 1. Build and Start the System
To build the worker images and start the system with an initial scale of **2 workers**:
```powershell
docker-compose up --build --scale worker=2
```
*   **`--build`**: Ensures the Dockerfile is processed and code changes are included.
*   **`--scale worker=2`**: Launches two identical worker containers.

### 2. Dynamic Scaling (Without Restarting)
While the system is running, you can scale the number of GPU nodes up or down instantly. Open a new terminal and run:
```powershell
# Scale up to 5 workers
docker-compose up -d --scale worker=5
```
HAProxy will automatically detect the new workers via Docker's internal DNS and add them to the cluster within seconds.

---

## Monitoring & Performance

### 1. HAProxy Stats Dashboard
Monitor the health of your nodes and real-time traffic distribution at:
**URL:** [http://localhost:8404/stats](http://localhost:8404/stats)

### 2. Run the Load Test (100 RPS)
Open a separate terminal and execute the simulation script:
```powershell
python main.py
```
*   The script will target 100 Requests Per Second for 10 seconds.
*   It will output a **Final Performance Report** including Actual RPS, Average Latency, and Success Rate.

---

## Simulation Logic
- **Concurrency Limit**: Each worker is limited to **2 concurrent "inferences"** via a Semaphore. Any additional requests will queue, simulating a busy GPU.
- **Inference Latency**: Each request is simulated to take **1.0 seconds** of compute time.
- **Hot Reloading**: The `worker.py` file is mounted as a volume. Any changes you save to the file will automatically trigger a reload inside the running containers.

---

## Stopping the System
To stop all containers and clean up the network:
```powershell
docker-compose down
```

---
*Created for CSE354: Distributed Computing – Ain Shams University*
