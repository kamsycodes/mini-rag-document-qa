
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_texts(self, texts: list[str]):
        """Convert a list of texts into numerical vectors."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
