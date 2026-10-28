
from src.embedder import Embedder
from src.vector_store import VectorStore



TEST_CASES = [
    {
        "question": "How does EdgeChain use smart contracts to manage IoT resources?",
        "expected_pages": [1, 2],
    },
    {
        "question": "What is the purpose of the credit-based resource management system?",
        "expected_pages": [3, 9],
    },
    {
        "question": "Why does EdgeChain use a permissioned blockchain?",
        "expected_pages": [2],
    },
    {
        "question": "What role does the edge server play in the EdgeChain architecture?",
        "expected_pages": [3, 6],
    },
    {
        "question": "What challenges of IoT systems does EdgeChain aim to address?",
        "expected_pages": [2],
    },
]



def evaluate():
    embedder = Embedder()
    store = VectorStore()

    hits = 0

    for case in TEST_CASES:
        vector = embedder.embed_texts([case["question"]])[0]
        results = store.search(vector, top_k=3)

        retrieved_pages = [
            result["page_number"] for result in results
        ]

        hit = any(
            page in case["expected_pages"]
            for page in retrieved_pages
        )

        hits += int(hit)

        print(f"Question: {case['question']}")
        print(f"Expected pages: {case['expected_pages']}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Hit@3: {hit}")

    hit_rate = hits / len(TEST_CASES)
    print(f"\nHit Rate@3: {hit_rate:.2%}")


if __name__ == "__main__":
    evaluate()
