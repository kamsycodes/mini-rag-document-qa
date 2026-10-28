# Mini RAG Document Q&A

A lightweight retrieval-augmented generation (RAG) application built in Python without LangChain. It extracts text from a PDF, creates overlapping text chunks, embeds the chunks locally, and stores them in ChromaDB. At question time, it retrieves relevant passages and uses an OpenAI model to generate an answer grounded in the document, with page references.

## Architecture

**Indexing (run when loading a document)**

`PDF → PyMuPDF text extraction → page-aware character chunking → Sentence Transformer embeddings → ChromaDB`

**Question answering (run per question)**

`Question → same embedding model → ChromaDB top-3 cosine-distance retrieval → distance filter → grounded GPT-4.1-mini answer + source pages`

The document's embeddings are stored locally, so the PDF does not have to be reindexed for every query. Retrieved text is sent to the OpenAI API when answer generation is requested.

## Technology stack

- **Python:** Application and CLI
- **PyMuPDF:** PDF text extraction with page metadata
- **Sentence Transformers (`all-MiniLM-L6-v2`):** Local 384-dimensional text embeddings
- **ChromaDB:** Persistent vector storage and cosine-distance search
- **OpenAI Responses API (`gpt-4.1-mini`):** Grounded answer generation
- **pytest:** Unit tests

## Project structure

```text
mini-rag/
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py       # Extract PDF text and page numbers
│   ├── chunker.py          # Split pages into overlapping chunks
│   ├── embedder.py         # Generate local text embeddings
│   ├── vector_store.py     # Store and search vectors in ChromaDB
│   ├── index_document.py   # Build the document index
│   ├── retrieve.py         # Inspect retrieved document passages
│   ├── generator.py        # Generate grounded answers via OpenAI
│   └── rag.py              # End-to-end question-answering CLI
├── tests/
│   ├── test_chunker.py
│   ├── evaluate_retrieval.py
│   ├── inspect_retrieval.py
│   └── evaluate_abstention.py
├── data/                    # Local PDFs; excluded from Git
├── chroma_db/               # Local vector index; excluded from Git
├── requirements.txt
└── README.md
```

## Setup (macOS)

Python 3.13 was used during development. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` in the repository root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

**Security:** Never commit `.env`, API credentials, personal PDFs, or the local vector database. API usage is billed separately from ChatGPT.

## Usage

1. Add a text-based PDF at `data/sample.pdf`.
2. Index it:

   ```bash
   python -m src.index_document
   ```

3. Inspect the retrieved passages without calling the LLM:

   ```bash
   python -m src.retrieve
   ```

4. Ask a question and generate a grounded answer:

   ```bash
   python -m src.rag
   ```

Example question: *How does EdgeChain use smart contracts to manage IoT resources?*

The application retrieves up to three chunks, filters them using a provisional cosine-distance cutoff of `0.60`, and generates an answer with source page references when evidence passes the filter. Otherwise, it returns an insufficient-information message.

## Testing and evaluation

Run unit tests:

```bash
python -m pytest -v
```

Run the retrieval development-set check:

```bash
python -m tests.evaluate_retrieval
```

Run the out-of-document query check:

```bash
python -m tests.evaluate_abstention
```

### Observed prototype results

Evaluated with a 14-page IEEE *EdgeChain* paper indexed into **187 chunks**, using 500-character chunks with 100-character overlap and 384-dimensional embeddings.

| Check | Observed result | Scope |
|---|---:|---|
| Page-level Hit Rate@3 | 5/5 (100%) | Small development set with provisional source-page labels informed by retrieved passages |
| Negative-query rejection | 5/5 (100%) | Five unrelated questions under a `0.60` cosine-distance cutoff |
| End-to-end answer generation | Worked | Manual example about smart contracts with page references |

**Interpretation:** These are smoke tests of a prototype, not estimates of production accuracy. Page-level Hit Rate@3 is not the same as full chunk-level recall. The preliminary source labels were selected with knowledge of retrieved results, so an independently labeled evaluation set is needed for reliable assessment.

## Engineering decisions

- **No LangChain:** Implements each stage directly to make inputs, outputs, and failure modes observable.
- **Local embeddings:** Avoids embedding API charges and reuses the same model for indexing and queries.
- **ChromaDB:** Simplifies persistent vector storage, metadata handling, and nearest-neighbor queries.
- **Overlapping character chunks:** Simple baseline, though it can split sentences or words.
- **Metadata-based citations:** Each chunk retains its PDF page number.
- **Evidence filtering:** Skips the LLM call when no result passes the provisional distance threshold.

## Limitations and future work

- Assumes selectable text in PDFs; scanned documents require OCR.
- PDF extraction can scramble reading order in multi-column research papers.
- Character chunking can break words or sentences and can create redundant results.
- The index is designed for one document; multi-document support requires document-specific chunk IDs and source metadata.
- The fixed `0.60` cosine-distance threshold is not calibrated and can reject relevant evidence or accept irrelevant evidence.
- An LLM-generated citation is not automatically proof that every claim is supported; citation/faithfulness evaluation is future work.
- Add independent relevance labels, expanded positive and negative queries, latency benchmarks, document-aware indexing, and possibly reranking before production use.

## Why RAG?

RAG retrieves document-specific information at inference time instead of requiring model fine-tuning to incorporate each document. It offers a practical route to source-grounded Q&A over documents that may change frequently. Fine-tuning may improve task behavior but does not replace evidence retrieval and citation checks.
