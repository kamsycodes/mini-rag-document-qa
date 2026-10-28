
from src.chunker import chunk_pages


def test_chunking_with_overlap():
    pages = [
        {
            "page_number": 1,
            "text": "ABCDEFGHIJ"
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=6,
        overlap=2
    )

    assert len(chunks) == 2
    assert chunks[0]["text"] == "ABCDEF"
    assert chunks[1]["text"] == "EFGHIJ"
    assert chunks[0]["page_number"] == 1


def test_invalid_overlap():
    pages = [{"page_number": 1, "text": "Hello"}]

    try:
        chunk_pages(pages, chunk_size=5, overlap=5)
        assert False, "Expected ValueError"
    except ValueError:
        pass
