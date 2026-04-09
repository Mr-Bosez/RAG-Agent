# OTC Medicine Assistant (RAG)

This repository contains a Streamlit-based retrieval-augmented generation (RAG) chatbot for OTC medicine reference data.

The app loads a local PDF (`Data/OTC_medicines_comprehensive_list_clear.pdf`), splits it into chunks, builds embeddings with `sentence-transformers`, stores them in a local FAISS index, and uses those retrieved passages to answer questions.

## Prerequisites

- Windows 10/11 or another OS with Python support
- Python 3.9+ installed
- Git (optional, if cloning the repository)

## Setup

1. Open PowerShell and navigate to the project folder:

   ```powershell
   cd "D:\RAG agent - Copy"
   ```

2. Create a virtual environment if one is not already present:

   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Upgrade pip and install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Configure the `.env` file in the repository root.

   The app expects a variable named `GROQ_API_KEY` in `.env`:

   ```text
   GROQ_API_KEY=gsk_your_groq_key_here
   ```

   If you already have a `.env` file, verify it is present at the project root and contains the correct key.

## Running the app

1. In the activated virtual environment, start Streamlit:

   ```powershell
   streamlit run app.py
   ```

2. Open the browser at the URL shown by Streamlit, typically:

   ```text
   http://localhost:8501
   ```

## How to use

- Upload or use the built-in OTC medicine PDF via the app's retrieval pipeline.
- Ask symptoms or a medicine name in the chat input.
- The app will search the indexed PDF and respond using retrieved passages.
- If a valid `GROQ_API_KEY` is configured, the app will route requests to the Groq OpenAI-compatible API.
- If no key is configured, the app still shows retrieved passages and fallback text without calling the LLM.

## Important files

- `app.py` – Streamlit UI and app logic
- `rag.py` – RAG pipeline, PDF loading, chunking, embedding, retrieval, and prompt generation
- `requirements.txt` – Python dependencies
- `.env` – local environment variables (gitignored)
- `Data/OTC_medicines_comprehensive_list_clear.pdf` – source OTC medicine PDF
- `.rag_cache/` – local cache directory for FAISS index and metadata

## Notes

- The `.env` file is loaded by `app.py` using `python-dotenv`.
- If the PDF is missing or cannot be read, the app will fail when building or loading the index.
- If Streamlit reports missing modules, re-run:

  ```powershell
  pip install -r requirements.txt
  ```

- If the app displays `Status: no key in .env`, confirm the key is set exactly as `GROQ_API_KEY=` and there are no extra spaces or blank lines.

## Rebuilding the index

Inside the app sidebar, click **Rebuild index from PDF** to force regeneration of the embeddings and FAISS index from the PDF source.

## Troubleshooting

- `ModuleNotFoundError`: activate the virtual environment and reinstall dependencies.
- `FileNotFoundError` for the PDF: verify `Data/OTC_medicines_comprehensive_list_clear.pdf` exists.
- If the app still does not read `.env`, ensure the file is saved in UTF-8 format and is located in the repository root.

---

Enjoy using the OTC Medicine Assistant!