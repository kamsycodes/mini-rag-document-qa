
from src.embedder import Embedder
from src.vector_store import VectorStore

MAX_DISTANCE = 0.60

NEGATIVE_QUESTIONS = [
    "Who won the FIFA World Cup in 2022?",
    "What is the capital of Brazil?",
    "How do I bake a chocolate cake?",
    "What is the population of Australia?",
    "Who wrote Romeo and Juliet?",
]


def main():
    embedder = Embedder()
    store = VectorStore()

    correctly_rejected = 0

    for question in NEGATIVE_QUESTIONS:
        vector = embedder.embed_texts([question])[0]
        results = store.search(vector, top_k=3)

        best_distance = min(
            result["distance"] for result in results
        )

        rejected = best_distance > MAX_DISTANCE

        correctly_rejected += int(rejected)

        print(f"\nQuestion: {question}")
        print(f"Best distance: {best_distance:.4f}")
        print(f"Correctly rejected: {rejected}")

    rejection_rate = correctly_rejected / len(NEGATIVE_QUESTIONS)

    print(
        f"\nNegative Query Rejection Rate: "
        f"{rejection_rate:.2%}"
    )


if __name__ == "__main__":
    main()
