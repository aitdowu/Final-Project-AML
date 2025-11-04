# RAG Course Notes Chatbot

A chatbot that answers questions about course PDFs using RAG (Retrieval-Augmented Generation). Built with open-source models for a final project in Applied Machine Learning.

## What it does

Upload your lecture slides, notes, or formula sheets as PDFs. The system extracts text, creates embeddings, and uses semantic search to find relevant chunks when you ask questions. It's set up to work with both text-based PDFs and scanned documents (OCR).

## Setup

### Install dependencies

First, install system dependencies for OCR (if you want OCR support):

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

Then install Python packages:

```bash
pip install -r requirements.txt
```

### Add your PDFs

1. Put your PDF files in the `data/` directory
2. (Optional) Add caption files for diagrams in `data/captions/` - name them like `Week5_captions.txt` for `Week5.pdf`

The caption files will be merged with the extracted text during processing.

## Running it

### 1. Process your PDFs

This extracts text, creates embeddings, and builds the vector database:

```bash
python src/ingest.py
```

Or if you need to use python3:

```bash
python3 src/ingest.py
```

If you run into HuggingFace cache permission issues, it'll create a local cache in the project directory.

### 2. Start the chat interface

```bash
python src/main.py
```

This launches a Gradio web interface. Open the URL it shows in your browser.

### 3. Optional: Run analysis notebooks

```bash
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/Evaluation.ipynb
```

## How it works

1. **PDF Processing**: Extracts text using PyPDF2, and optionally OCR for scanned docs
2. **Chunking**: Splits documents into 800-character chunks with 100-character overlap
3. **Embeddings**: Uses `all-MiniLM-L6-v2` from sentence-transformers (384-dim vectors)
4. **Vector Store**: FAISS for fast similarity search
5. **Retrieval**: Finds top-k most similar chunks to your query
6. **Generation**: Currently uses a placeholder - needs an actual LLM implementation

## Technical details

- **Embedding model**: `all-MiniLM-L6-v2` - good balance of speed and quality, runs locally
- **Vector DB**: FAISS IndexFlatIP (exact cosine similarity search)
- **Chunk size**: 800 chars, 100 char overlap
- **Interface**: Gradio

The embedding model choice is pretty standard - it's fast, works well for semantic search, and doesn't need a GPU. FAISS is straightforward for exact search (could upgrade to approximate search if you have millions of docs).

## Project structure

```
rag_course_notes_chatbot/
├── data/              # Your PDFs go here (not in git)
├── data/captions/     # Optional caption files
├── notebooks/         # EDA and evaluation notebooks
├── src/
│   ├── ingest.py      # Processes PDFs and creates embeddings
│   ├── main.py        # Chat interface
│   └── utils.py       # Helper functions
├── db/                # Vector database (generated)
├── results/           # Outputs from notebooks
└── requirements.txt
```

## Current status

- PDF text extraction and OCR
- Caption merging
- Embedding generation and FAISS storage
- Semantic search and retrieval
- Gradio chat interface
- LLM generation is still a placeholder - needs actual model integration

## TODO / Future work

- Replace the placeholder LLM with an actual open-source model (Llama 2, Mistral, etc.)
- Fine-tune embeddings on domain-specific data
- Add query expansion for better retrieval
- Hybrid search (semantic + keyword)
- User feedback mechanism

## Dependencies

Main packages:
- `sentence-transformers` - embeddings
- `faiss-cpu` - vector search
- `gradio` - web interface
- `PyPDF2` - PDF text extraction
- `pdf2image` + `pytesseract` - OCR
- `langchain` - text chunking

See `requirements.txt` for the full list.

