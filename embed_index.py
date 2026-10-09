
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"
DB_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "bigbrains_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 35
BATCH_SIZE = 100


# ==================== TEXT CHUNKING ====================

def split_into_chunks(text):
    """Split text into overlapping word-based chunks."""

    words = text.split()
    chunks = []

    if not words:
        return chunks

    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==================== LOAD KNOWLEDGE BASE ====================

def load_knowledge_base():
    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KB_PATH}"
        )

    with KB_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "knowledge_base.json must contain a JSON object."
        )

    documents = data.get("documents", [])

    if not isinstance(documents, list) or not documents:
        raise ValueError(
            "No documents found in knowledge_base.json."
        )

    return documents


# ==================== PREPARE CHUNKS ====================

def prepare_records(documents):
    records = []

    for doc_index, doc in enumerate(documents):
        if not isinstance(doc, dict):
            print(
                f"Skipping invalid document at index {doc_index}."
            )
            continue

        title = str(
            doc.get("title") or "Untitled"
        ).strip()

        category = str(
            doc.get("category") or "general"
        ).strip()

        url = str(
            doc.get("url") or ""
        ).strip()

        content = str(
            doc.get("content") or ""
        ).strip()

        if not content:
            print(f"Skipping empty document: {title}")
            continue

        chunks = split_into_chunks(content)

        for chunk_index, chunk in enumerate(chunks):
            # Include source context to improve semantic retrieval.
            searchable_text = (
                f"Title: {title}\n"
                f"Category: {category}\n"
                f"Content: {chunk}"
            )

            records.append({
                "id": f"doc_{doc_index}_chunk_{chunk_index}",
                "text": searchable_text,
                "metadata": {
                    "title": title,
                    "category": category,
                    "url": url,
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                },
            })

    if not records:
        raise ValueError(
            "No usable text chunks were generated."
        )

    return records


# ==================== BUILD VECTOR INDEX ====================

def main():
    print("Loading Big Brains knowledge base...")

    documents = load_knowledge_base()
    records = prepare_records(documents)

    print(f"Source documents: {len(documents)}")
    print(f"Prepared text chunks: {len(records)}")

    DB_PATH.mkdir(parents=True, exist_ok=True)

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Rebuild the index so removed or updated source documents
    # do not leave stale vectors in the collection.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Previous collection removed.")
    except Exception:
        print("No previous collection to remove.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("\nGenerating embeddings and storing vectors...")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        texts = [
            record["text"]
            for record in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        collection.add(
            ids=[
                record["id"]
                for record in batch
            ],
            documents=texts,
            metadatas=[
                record["metadata"]
                for record in batch
            ],
            embeddings=embeddings.tolist(),
        )

        processed = min(
            start + len(batch),
            len(records),
        )

        print(
            f"Indexed {processed}/{len(records)} chunks"
        )

    # ==================== VERIFY INDEX ====================

    print("\nVerifying the saved vector database...")

    stored_count = collection.count()

    if stored_count != len(records):
        raise RuntimeError(
            f"Index verification failed: expected {len(records)} "
            f"vectors, found {stored_count}."
        )

    test_query = (
        "What is the duration and participation fee "
        "of the Generative AI internship?"
    )

    query_embedding = model.encode(
        test_query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, stored_count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    print("\n========== RETRIEVAL TEST ==========")

    for index, result_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]
        document = results["documents"][0][index]

        print(f"\nResult {index + 1}")
        print(f"ID: {result_id}")
        print(f"Title: {metadata.get('title', 'Unknown')}")
        print(f"Category: {metadata.get('category', 'general')}")
        print(f"URL: {metadata.get('url', '')}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Text preview: {document[:500]}")

    print("\n========== PHASE 3 COMPLETE ==========")
    print(f"Source documents: {len(documents)}")
    print(f"Generated chunks: {len(records)}")
    print(f"Stored vectors: {stored_count}")
    print(f"Embedding dimensions: {len(query_embedding)}")
    print(f"Database: {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print("======================================")


if __name__ == "__main__":
    main()