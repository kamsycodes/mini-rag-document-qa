
from src.embedder import Embedder
from src.vector_store import VectorStore


def main():
    print("Loading embedding model...")
    embedder = Embedder()

    print("Connecting to vector database...")
    store = VectorStore()

    print(f"Indexed chunks: {store.count()}")

    if store.count() == 0:
        print("No indexed documents found. Run indexing first.")
        return

    question = input("\nAsk a question about EdgeChain: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nGenerating question embedding...")
    query_embedding = embedder.embed_texts([question])[0]

    print("Searching document...\n")
    results = store.search(query_embedding, top_k=3)

    for rank, result in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"RESULT {rank}")
        print(f"Page: {result['page_number']}")
        print(f"Cosine distance: {result['distance']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"\n{result['text']}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
