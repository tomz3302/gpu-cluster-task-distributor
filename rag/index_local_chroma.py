from pathlib import Path
from typing import List
import re

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "project_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"




def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
    """
    Splits text into overlapping chunks.
    Chunks are character-based for simplicity.
    """

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def safe_id(text: str) -> str:
    """
    Chroma IDs should be simple strings.
    This converts paths like Lectures/Lecture1.txt into safe IDs.
    """

    text = text.replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^a-zA-Z0-9_.-]", "_", text)
    return text


def load_knowledge_files():
    """
    Recursively loads all .txt files inside rag/knowledge/.
    Skips Prompt.txt because that is a prompt template, not searchable knowledge.
    """

    files = sorted(KNOWLEDGE_DIR.rglob("*.txt"))

    documents = []

    for file_path in files:
        if file_path.name.lower() == "prompt.txt":
            print(f"Skipping prompt file: {file_path}")
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            print(f"Skipping empty file: {file_path}")
            continue

        relative_source = file_path.relative_to(KNOWLEDGE_DIR).as_posix()

        documents.append({
            "source": relative_source,
            "text": text,
        })

    return documents


def build_index(reset: bool = True):
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connecting to local persistent ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if reset:
        print("Resetting collection...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Loading knowledge files...")
    docs = load_knowledge_files()

    if not docs:
        raise RuntimeError(f"No .txt knowledge files found in {KNOWLEDGE_DIR}")

    ids = []
    texts = []
    metadatas = []

    print("Chunking documents...")

    for doc in docs:
        source = doc["source"]
        chunks = chunk_text(doc["text"])

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{safe_id(source)}_chunk_{chunk_index}"

            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "source": source,
                "chunk_id": chunk_index,
            })

    print(f"Total chunks: {len(texts)}")

    print("Creating embeddings locally...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print("Writing chunks and embeddings to local ChromaDB...")

    batch_size = 500

    for start in tqdm(range(0, len(texts), batch_size)):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print("Done.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Total indexed chunks: {collection.count()}")
    print(f"Local ChromaDB path: {CHROMA_PATH}")


if __name__ == "__main__":
    build_index(reset=True)