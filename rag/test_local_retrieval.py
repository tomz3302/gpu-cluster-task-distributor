from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "project_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    query = "Explain what load balancing is in one short paragraph."

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connecting to local ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    print("Embedding query...")
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    print("Searching...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print("\nQuery:")
    print(query)

    print("\nTop results:")

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print("-" * 80)
        print(f"Rank: {i + 1}")
        print(f"Source: {metadata['source']}")
        print(f"Chunk: {metadata['chunk_id']}")
        print(f"Distance: {distance:.4f}")
        print()
        print(document[:1000])


if __name__ == "__main__":
    main()