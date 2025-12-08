# AML Notes RAG Chatbot

Built a RAG chatbot for my Applied Machine Learning course notes. Basically you upload PDFs, ask questions, and it gives you answers with citations. Pretty useful for studying.

## Data Sources

I used the course notes from my Applied Machine Learning class (weeks 1-9). They're all PDF files with lecture notes and slides.

- PDFs are in the `data/` folder
- I used PyPDF2 to extract text, and pytesseract for OCR on scanned pages (some of the PDFs were scanned)
- There's also a `data/captions/` folder where I put text descriptions of diagrams - these get merged in during processing

The notes cover stuff like:
- ML basics
- Neural networks
- Model evaluation
- Feature engineering
- Optimization
- Other AML topics


Then install Python packages:

```bash
pip install -r requirements.txt
```

To add your PDFs:
1. Put them in `data/`
2. If you have caption files for diagrams, put them in `data/captions/` - name them like `Week5_captions.txt` for `Week5.pdf`

## Usage

### Process PDFs

Run this to extract text and build the vector database:

```bash
python Pipeline/main_pipeline.py
```

First time it'll download the embedding model (Snowflake Arctic Embed M v2.0) - takes a bit but it caches it.

### Start the chatbot

```bash
# from project root
export HF_TOKEN="your_token_here"           # only if model needs it
export HF_HOME="$(pwd)/.cache/huggingface"  # keeps cache local
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"
PYTHONPATH=. python3 Pipeline/rag_chatbot.py
```

First run downloads Qwen2-1.5B-Instruct. After it starts, open the URL it prints in your browser.

### Reset database

If you want to reprocess everything or change the embedding model:

```bash
./reset_database.sh
```

Or just do it manually:
```bash
rm -rf db/
python Pipeline/main_pipeline.py
```

## Reproducing Results

To get the same results I got:

1. Install dependencies (see above)

2. Process the PDFs:
   ```bash
   python Pipeline/main_pipeline.py
   ```
   This builds the vector database in `db/`.

3. Run the evaluation notebook:
   - Open `Notebook/Evaluation.ipynb` in Jupyter
   - Run all the cells
   - It saves results to:
     - `results/evaluation_performance_table.csv`
     - `results/evaluation_report.txt`
     - `figures/evaluation_performance_metrics.png`

4. To view results:
   - Load CSV: `pd.read_csv('results/evaluation_performance_table.csv')`
   - Check report: `cat results/evaluation_report.txt`
   - Figures are in `figures/`

All the plots come from the CSV files so you can reproduce them.

## How it works

Pretty straightforward:
1. Extract text from PDFs using PyPDF2 (OCR for scanned stuff)
2. Split into chunks (800 chars, 100 char overlap - seemed like good defaults)
3. Generate embeddings with Snowflake Arctic Embed M v2.0 (768 dims)
4. Store in FAISS for fast similarity search
5. When you ask a question, find top 5 most similar chunks
6. Use Qwen2-1.5B-Instruct to generate an answer from those chunks

## Technical details

- Embedding: Snowflake Arctic Embed M v2.0 (768 dims)
- LLM: Qwen2-1.5B-Instruct (1.5B params - small enough to run on my laptop)
- Vector DB: FAISS IndexFlatIP
- Chunk size: 800 chars, 100 overlap
- Interface: Gradio (super easy to use)

Everything runs locally.

## Project structure

```
Final Project AML/
├── Models/                    # Model definitions
│   ├── embedding_model.py     # Embedding generator (Snowflake)
│   ├── llm_model.py           # LLM wrapper (Qwen2)
│   └── vector_store.py         # FAISS vector store
│
├── Trainer/                    # Training and evaluation
│   └── eval_epoch.py          # Evaluation functions and metrics
│
├── Pipeline/                   # Main pipeline orchestration
│   ├── main_pipeline.py       # PDF processing and database creation
│   └── rag_chatbot.py         # RAG chatbot interface
│
├── Utils/                      # Utility functions
│   ├── pdf_processor.py       # PDF text extraction and chunking
│   ├── helper_functions.py    # RAG helper functions
│   └── extract_images.py      # Image extraction utility
│
├── Notebook/                   # Analysis notebooks
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   └── Evaluation.ipynb       # Model evaluation
│
├── data/                       # Input data
│   ├── *.pdf                  # PDF course materials
│   └── captions/              # Optional caption files
│
├── db/                         # Vector database (generated)
│   ├── vector_store.index     # FAISS index
│   ├── vector_store.metadata  # Chunk metadata
│   └── vector_store.summary.json  # Database summary
│
├── results/                    # Evaluation results (CSV, TXT)
│   ├── evaluation_performance_table.csv
│   └── evaluation_report.txt
│
├── figures/                    # Generated plots and visualizations
│   ├── eda_text_length_analysis.png
│   └── evaluation_performance_metrics.png
│
├── extracted_images/           # Extracted images from PDFs
├── reset_database.sh           # Script to reset database
└── requirements.txt            # Python dependencies
```

## Dependencies

Main stuff I used:
- `transformers` + `torch` - for the models
- `faiss-cpu` - vector search (super fast)
- `gradio` - web UI
- `PyPDF2` - extract text from PDFs
- `pdf2image` + `pytesseract` - OCR
- `langchain` - text chunking
- `sentence-transformers` - backup embedding support

Full list in `requirements.txt`.
