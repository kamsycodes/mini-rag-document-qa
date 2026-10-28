
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.generator import AnswerGenerator


MAX_DISTANCE = 0.60


def main():
    embedder = Embedder()
    store = VectorStore()

    if store.count() == 0:
        print("No documents indexed. Run indexing first.")
        return

    question = input("Ask a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    query_vector = embedder.embed_texts([question])[0]

    results = store.search(query_vector, top_k=3)

    relevant_chunks = [
        chunk for chunk in results
        if chunk["distance"] <= MAX_DISTANCE
    ]

    if not relevant_chunks:
        print("\nI couldn't find enough information in the document.")
        return

    generator = AnswerGenerator()

    answer = generator.generate(question, relevant_chunks)

    print("\nANSWER")
    print(answer)

    print("\nRETRIEVED SOURCES")
    for chunk in relevant_chunks:
        print(
            f"Page {chunk['page_number']} | "
            f"Distance: {chunk['distance']:.4f}"
        )


if __name__ == "__main__":
    main()
