# RAG-Enabled GPU Worker Setup Guide

This guide explains how to set up and run one **RAG-enabled GPU worker node** on a teammate's PC.

Each worker runs:

```text
FastAPI Worker API
    ↓
Local ChromaDB vector database
    ↓
Local RAG retriever
    ↓
Local Ollama LLM
    ↓
Response
```

The load balancer/master scheduler is intentionally **not included** in this guide. The goal is only to make each teammate's PC work as an individual worker node.

---

## 1. What each teammate needs to install

Install these on every worker PC:

1. **Python 3.10+ or 3.11**
2. **Git**
3. **Ollama**
4. **ZeroTier One**
5. Project Python dependencies


---

## 2. Required project folder structure

After cloning or copying the project, the folder should look like this:

```text
gpu-cluster-task-distributor/
├── rag/
│   ├── __init__.py
│   ├── chroma_db/
│   ├── knowledge/
│   │   ├── Lectures/
│   │   │   ├── Lecture1.txt
│   │   │   ├── Lecture2.txt
│   │   │   └── ...
│   │   ├── Questions/
│   │   │   ├── Lecture1-long.txt
│   │   │   ├── Lecture1-short.txt
│   │   │   └── ...
│   │   └── Prompt.txt
│   ├── index_local_chroma.py
│   ├── retriever.py
│   └── test_local_retrieval.py
├── workers/
│   ├── __init__.py
│   └── gpu_worker_api.py
├── client/
│   └── worker_load_test.py
└── requirements.txt
```

Important notes:

- `rag/knowledge/` contains the source `.txt` files that RAG searches.
- `rag/chroma_db/` is the local ChromaDB vector database folder.
- `rag/index_local_chroma.py` builds the vector database from the knowledge files.
- `rag/retriever.py` retrieves relevant context from ChromaDB.
- `workers/gpu_worker_api.py` runs the FastAPI worker.
- `client/worker_load_test.py` can test one worker directly.

---



## 4. Create and activate a Python virtual environment

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

If activation worked, you should see `(.venv)` at the start of your terminal line.

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

## 5. Install Python dependencies

Run:

```powershell
pip install chromadb sentence-transformers tqdm fastapi uvicorn ollama pydantic httpx nvidia-ml-py
```

Optional: if the project has a `requirements.txt`, you can use:

```powershell
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
chromadb
sentence-transformers
tqdm
fastapi
uvicorn
ollama
pydantic
httpx
nvidia-ml-py
```

---

## 6. Install Ollama

### Windows

In PowerShell:

```powershell
irm https://ollama.com/install.ps1 | iex
```

After installation, close and reopen PowerShell.

Check Ollama:

```powershell
ollama --version
```

Pull the small model used for the worker:

```powershell
ollama pull smollm:135m
```

Test the model:

```powershell
ollama run smollm:135m
```

Type:

```text
Explain load balancing in one short paragraph.
```

To exit:

```text
/bye
```

---

## 7. Prepare the knowledge files

Make sure the knowledge files exist under:

```text
rag/knowledge/
```

Example:

```text
rag/knowledge/Lectures/Lecture1.txt
rag/knowledge/Lectures/Lecture2.txt
rag/knowledge/Questions/Lecture1-short.txt
rag/knowledge/Questions/Lecture1-long.txt
```

---

## 8. Build the local ChromaDB index

From the project root, run:

```powershell
python rag\index_local_chroma.py
```

Expected output should include something like:

```text
Loading embedding model...
Connecting to local ChromaDB...
Resetting collection...
Loading knowledge files...
Chunking documents...
Total chunks: ...
Creating embeddings locally...
Writing chunks and embeddings to local ChromaDB...
Done.
Collection name: project_knowledge
Total indexed chunks: ...
Local ChromaDB path: ...\rag\chroma_db
```

This creates or updates the local vector database inside:

```text
rag/chroma_db/
```

Run this script again whenever knowledge files are added or changed.

---

## 9. Test local RAG retrieval

Run:

```powershell
python rag\test_local_retrieval.py
```

Expected result:

- It should print the query.
- It should print retrieved chunks from the knowledge base.
- It should print retrieval time.
- It should print the final RAG prompt.

If this fails, fix the ChromaDB indexing before starting the worker.

---

## 10. Start Ollama

Open a new PowerShell window.

Run:

```powershell
ollama serve
```

Leave this terminal open.

If Ollama says the address is already in use, that usually means Ollama is already running in the background. That is fine.

---

## 11. Start the RAG GPU worker

Open another PowerShell window in the project root:

```powershell
cd "C:\path\to\gpu-cluster-task-distributor"
.venv\Scripts\activate
```

Set the worker name:

```powershell
$env:WORKER_NAME="teammate-worker-1"
```

Set the inference concurrency:

```powershell
$env:MAX_CONCURRENT_INFERENCE="1"
```

Optional: set a different Ollama model:

```powershell
$env:OLLAMA_MODEL="smollm:135m"
```

Start the worker:

```powershell
uvicorn workers.gpu_worker_api:app --host 0.0.0.0 --port 8000
```

Important:

```text
--host 0.0.0.0
```

This allows other devices on the ZeroTier network to reach the worker.

---

## 12. Test the worker locally

Open another PowerShell window.

Test `/health`:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
```

Expected response:

```json
{
  "status": "ok",
  "worker_name": "teammate-worker-1",
  "model": "smollm:135m",
  "rag_enabled": true,
  "max_concurrent_inference": 1
}
```

Test `/generate`:

```powershell
$body = @{
  id = 1
  query = "What is round robin load balancing?"
  max_tokens = 64
  temperature = 0.2
  top_k = 3
  use_rag = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8000/generate" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

Expected response should include:

```text
result
used_rag
sources
queue_time
retrieval_time
inference_time
total_latency
```

---

## 13. Install ZeroTier

ZeroTier lets workers communicate over the internet using private VPN IPs.

Install ZeroTier on Windows:

```powershell
winget install --id=ZeroTier.ZeroTierOne -e
```

After installation, close and reopen PowerShell.

Check it:

```powershell
zerotier-cli -v
```

If this command fails, try running PowerShell as Administrator.

---

## 14. Join the team ZeroTier network

The project owner should provide a ZeroTier Network ID.

Join it:

```powershell
zerotier-cli join a581878f7d0e8bef
```


Expected output:

```text
200 join OK
```

Then tell the project owner to authorize your device in ZeroTier Central.

---

## 15. Find your ZeroTier IP

After authorization, run:

```powershell
zerotier-cli listnetworks
```

Look for your managed IP. It may look like:

```text
10.147.17.101
```

This is your worker's ZeroTier IP.

Your worker URL will be:

```text
http://YOUR_ZEROTIER_IP:8000
```

Example:

```text
http://10.147.17.101:8000
```

---

## 16. Allow port 8000 through Windows Firewall

Run PowerShell as Administrator.

```powershell
New-NetFirewallRule `
  -DisplayName "GPU RAG Worker API Port 8000" `
  -Direction Inbound `
  -Action Allow `
  -Protocol TCP `
  -LocalPort 8000
```

This allows other ZeroTier devices to call your worker API.

---

## 17. Test the worker over ZeroTier

From another device that is also connected to the same ZeroTier network, run:

```powershell
Invoke-RestMethod -Uri "http://YOUR_ZEROTIER_IP:8000/health" -Method GET
```

Example:

```powershell
Invoke-RestMethod -Uri "http://10.147.17.101:8000/health" -Method GET
```

Then test `/generate`:

```powershell
$body = @{
  id = 1
  query = "What is round robin load balancing?"
  max_tokens = 64
  temperature = 0.2
  top_k = 3
  use_rag = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://YOUR_ZEROTIER_IP:8000/generate" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://10.147.17.101:8000/generate" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

If this works, the worker is reachable over the internet through ZeroTier.

---

## 18. Run a small worker benchmark

Before running a large test, use a small benchmark.

Set the worker URL:

```powershell
$env:WORKER_URL="http://YOUR_ZEROTIER_IP:8000/generate"
```

Run:

```powershell
python client\worker_load_test.py
```

For first testing, keep the benchmark small inside `client/worker_load_test.py`:

```python
test_plan = [
    (1, 3),
    (2, 5),
    (4, 10),
]
```

After confirming the worker is stable, increase the test sizes gradually.

---

## 19. Recommended worker settings

Start with:

```powershell
$env:MAX_CONCURRENT_INFERENCE="1"
```

Then test:

```powershell
$env:MAX_CONCURRENT_INFERENCE="2"
```

Then:

```powershell
$env:MAX_CONCURRENT_INFERENCE="4"
```

Higher is not always better. More concurrent inference requests may increase GPU/CPU contention and make responses slower.

Use benchmark results to decide the best setting.

---

## 20. What each teammate should report to the load balancer team

Each teammate should send:

```text
Worker name:
ZeroTier IP:
Worker URL:
Model:
MAX_CONCURRENT_INFERENCE:
Average latency:
P95 latency:
Throughput:
Notes/errors:
```

Example:

```text
Worker name: amr-worker-1
ZeroTier IP: 10.147.17.101
Worker URL: http://10.147.17.101:8000
Model: smollm:135m
MAX_CONCURRENT_INFERENCE: 1
Throughput: 6.5 req/s
Average latency: 4.7s
P95 latency: 4.8s
Notes: RAG enabled, top_k=3
```

---

## 21. Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'rag'`

Make sure you are running commands from the project root.

Correct:

```powershell
python rag\test_local_retrieval.py
uvicorn workers.gpu_worker_api:app --host 0.0.0.0 --port 8000
```

Also make sure these files exist:

```text
rag/__init__.py
workers/__init__.py
```

---

### Problem: Chroma collection does not exist

Run:

```powershell
python rag\index_local_chroma.py
```

Then test:

```powershell
python rag\test_local_retrieval.py
```

---

### Problem: Ollama connection error

Make sure Ollama is running:

```powershell
ollama serve
```

Also make sure the model is downloaded:

```powershell
ollama list
```

If missing:

```powershell
ollama pull smollm:135m
```

---

### Problem: Another device cannot reach the worker

Check:

1. Worker was started with `--host 0.0.0.0`
2. Both devices are joined to the same ZeroTier network
3. Both devices are authorized in ZeroTier Central
4. You are using the ZeroTier IP, not the Wi-Fi IP
5. Windows Firewall allows TCP port `8000`
6. The worker is running on the target machine

Test on the worker machine:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
```

Then test using its own ZeroTier IP:

```powershell
Invoke-RestMethod -Uri "http://YOUR_ZEROTIER_IP:8000/health" -Method GET
```

Then test from another device.

---

### Problem: Benchmark seems stuck

RAG is slower than plain LLM inference because each request does retrieval before generation.

Start with a tiny test plan:

```python
test_plan = [
    (1, 1),
    (1, 3),
    (2, 5),
]
```

Also reduce:

```python
MAX_TOKENS = 64
```

---

## 22. Running the GUI Chat Interface

We have built a simple, sleek chat GUI to interact with the cluster!

### Prerequisites:
Make sure your HAProxy (load balancer) is running on `http://localhost:8080/generate` before starting the GUI.

### Steps to Use:
1. Open a PowerShell/Terminal window in the project root.
2. Activate your virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
3. Run the GUI Python file:
   ```powershell
   python Simple_Gui.py
   ```
4. A dark-themed chat interface will appear.
5. In the message box at the bottom, type your question and hit **Send** or press **Enter**.

### Important Note on Answers:
The HAProxy configuration automatically relies on the backend GPU workers, which process questions using **RAG (Retrieval-Augmented Generation)**. 
**This means that the AI engine strictly parses and retrieves answers based ONLY on the provided local `.txt` documents present in the `rag/knowledge/` directory.** It will provide sources and metrics as part of the interface output.

---

