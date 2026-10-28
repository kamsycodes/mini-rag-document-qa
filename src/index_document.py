
from src.pdf_loader import load_pdf
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.vector_store import VectorStore


def main():
    pdf_path = "data/sample.pdf"

    print("Loading PDF...")
    pages = load_pdf(pdf_path)

    print("Chunking document...")
    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=100
    )

    print(f"Generated {len(chunks)} chunks")

    print("Generating embeddings...")
    embedder = Embedder()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    print(f"Embedding matrix: {embeddings.shape}")

    print("Storing vectors in ChromaDB...")
    store = VectorStore()
    store.add_chunks(chunks, embeddings)

    print(f"Total stored chunks: {store.count()}")
    print("Indexing complete!")


if __name__ == "__main__":
    main()

