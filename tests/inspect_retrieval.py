
from src.embedder import Embedder
from src.vector_store import VectorStore


QUESTIONS = [
    "How does EdgeChain use smart contracts to manage IoT resources?",
    "What is the purpose of the credit-based resource management system?",
    "Why does EdgeChain use a permissioned blockchain?",
    "What role does the edge server play in the EdgeChain architecture?",
    "What challenges of IoT systems does EdgeChain aim to address?",
]


def main():
    embedder = Embedder()
    store = VectorStore()

    for question in QUESTIONS:
        vector = embedder.embed_texts([question])[0]
        results = store.search(vector, top_k=3)

        print(f"\nQUESTION: {question}")

        for result in results:
            print(
                f"\nPage: {result['page_number']} | "
                f"Distance: {result['distance']:.4f}"
            )
            print(result["text"][:350])
            print("-" * 50)


if __name__ == "__main__":
    main()
