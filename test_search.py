
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "bigbrains_knowledge"


def main():
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    model = SentenceTransformer(MODEL_NAME)

    print("Big Brains Semantic Search Test")
    print(f"Indexed chunks: {collection.count()}")

    while True:
        question = input(
            "\nEnter a question (or type exit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        for index, document in enumerate(
            results["documents"][0],
            start=1,
        ):
            metadata = results["metadatas"][0][index - 1]
            distance = results["distances"][0][index - 1]

            print(f"\nResult {index}")
            print("Title:", metadata.get("title"))
            print("Category:", metadata.get("category"))
            print("URL:", metadata.get("url"))
            print("Cosine distance:", round(distance, 4))
            print("Content:", document[:700])


if __name__ == "__main__":
    main()