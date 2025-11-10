# AML Notes RAG Chatbot

RAG chatbot for Applied Machine Learning course notes. Upload PDFs, ask questions, get answers with source citations.

## Setup

### Install dependencies

Install system dependencies for OCR (optional):

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

Install Python packages:

```bash
pip install -r requirements.txt
```

### Add PDFs

1. Put PDF files in the `data/` directory
2. Optionally add caption files for diagrams in `data/captions/` - name them like `Week5_captions.txt` for `Week5.pdf`

Caption files are merged with extracted text during processing.

## Usage

### Process PDFs

Extract text, create embeddings, and build the vector database:

```bash
python3 src/ingest.py
```

First run downloads the embedding model (Snowflake Arctic Embed M v2.0). The model is cached locally.

### Start the chat interface

```bash
python3 src/main.py
```

First run downloads the Qwen2-1.5B-Instruct model. Open the URL shown in your browser.

### Reset database

If you change the embedding model or want to reprocess all PDFs:

```bash
./reset_database.sh
```

Or manually:

```bash
rm -rf db/
python3 src/ingest.py
```

## How it works

1. PDF processing: Extracts text using PyPDF2, with optional OCR for scanned documents
2. Chunking: Splits documents into 800-character chunks with 100-character overlap
3. Embeddings: Uses Snowflake Arctic Embed M v2.0 (768-dim vectors) with mean pooling
4. Vector store: FAISS for similarity search
5. Retrieval: Finds top-5 most similar chunks to queries
6. Generation: Uses Qwen2-1.5B-Instruct to generate answers from retrieved context

## Technical details

- Embedding model: Snowflake Arctic Embed M v2.0 (768 dimensions)
- LLM: Qwen2-1.5B-Instruct (1.5B parameters)
- Vector DB: FAISS IndexFlatIP (cosine similarity)
- Chunk size: 800 chars, 100 char overlap
- Interface: Gradio

The embedding model runs locally and works well for semantic search. Qwen2 is small enough to run on CPU, though it's faster with a GPU.

## Project structure

```
rag_course_notes_chatbot/
├── data/              # PDFs go here
├── data/captions/     # Optional caption files
├── notebooks/         # Analysis notebooks
├── src/
│   ├── ingest.py      # Process PDFs and create embeddings
│   ├── main.py        # Chat interface
│   └── utils.py       # Helper functions
├── db/                # Vector database (generated)
├── reset_database.sh  # Script to reset database
└── requirements.txt
```

## Dependencies

Main packages:
- `transformers` + `torch` - Qwen2 LLM and Snowflake embeddings
- `faiss-cpu` - Vector search
- `gradio` - Web interface
- `PyPDF2` - PDF text extraction
- `pdf2image` + `pytesseract` - OCR
- `langchain` - Text chunking
- `sentence-transformers` - Fallback embedding support

See `requirements.txt` for the full list.
