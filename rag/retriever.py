import time
from pathlib import Path
from typing import Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "project_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


print("Loading local embedding model for retrieval...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Connecting to local ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(COLLECTION_NAME)


def retrieve_context(query: str, top_k: int = 3) -> Dict[str, Any]:
    start = time.perf_counter()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieval_time = time.perf_counter() - start

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_blocks = []

    for i, doc in enumerate(docs):
        source = metadatas[i].get("source", "unknown")
        chunk_id = metadatas[i].get("chunk_id", "unknown")
        distance = distances[i]

        context_blocks.append(
            f"[Source: {source}, chunk: {chunk_id}, distance: {distance:.4f}]\n{doc}"
        )

    context = "\n\n".join(context_blocks)

    return {
        "context": context,
        "sources": metadatas,
        "distances": distances,
        "retrieval_time": retrieval_time,
    }


def build_rag_prompt(query: str, context: str) -> str:
    return f"""
You are an assistant for a distributed computing course project.

Use only the context below to answer the question.
If the context does not contain enough information, say:
"The knowledge base does not contain enough information."

Context:
{context}

Question:
{query}

Answer:
""".strip()