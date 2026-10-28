
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Chunk size and overlap are measured in characters.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"page_{page_number}_chunk_{len(chunks)}",
                    "page_number": page_number,
                    "text": chunk_text
                })

            if end == len(text):
                break

            start += chunk_size - overlap

    return chunks
