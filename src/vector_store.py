
import chromadb


class VectorStore:
    def __init__(self, db_path="chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="edgechain_docs",
            configuration={"hnsw": {"space": "cosine"}}
        )

    def add_chunks(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match"
            )

        self.collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {"page_number": chunk["page_number"]}
                for chunk in chunks
            ],
            embeddings=embeddings.tolist()
        )

    def count(self):
        return self.collection.count()
    

    def search(self, query_embedding, top_k=3):
        """Retrieve the most similar document chunks."""

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []

        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "page_number": results["metadatas"][0][i]["page_number"],
                "distance": results["distances"][0][i]
            })

        return retrieved_chunks

