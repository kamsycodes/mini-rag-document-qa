
from pathlib import Path
import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF, preserving page numbers.
    
    Returns:
        A list of dictionaries containing page_number and text.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    return pages


if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF: ").strip()

    try:
        pages = load_pdf(pdf_path)

        print(f"\nSuccessfully extracted {len(pages)} pages with text.")

        if pages:
            print("\nFirst page preview:")
            print(pages[0]["text"][:500])

    except (FileNotFoundError, pymupdf.FileDataError) as error:
        print(f"Error: {error}")
