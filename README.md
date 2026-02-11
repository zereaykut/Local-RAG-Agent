# Local AI Agent with RAG

A modular Python implementation of a Local AI Agent capable of **Retrieval Augmented Generation (RAG)**. This application runs entirely on your local machine using [Ollama](https://ollama.com/), ensuring data privacy and offline capability.

It allows you to chat with your own documents (PDFs, TXT or CSV files) by embedding them into a local vector database.

## 🌟 Features

- **100% Local**: Uses Ollama for LLM inference (Llama3, Mistral, etc.) and embeddings.
- **RAG Capable**: automatically ingests, splits, and indexes documents from a `data` folder.
- **Persistent Memory**: Uses ChromaDB to save embeddings so you don't have to re-index every time.
- **Modular Design**: Separation of concerns between configuration, RAG logic, and Agent interaction.

## 📂 Project Structure

```text
├── data/
│   └── source_docs/
│       └── realistic_restaurant_reviews.csv  <-- Place it here
├── src/                    
│   ├── config.py           # Settings and Environment variables
│   ├── rag_engine.py       # Document processing & Vector Store logic
│   └── bot_agent.py        # LLM interaction logic
├── .env                    # Configuration file (Model names, paths)
├── main.py                 # Entry point to run the application
└── requirements.txt        # Python dependencies
```

## 🚀 Prerequisites

1. Python 3.10+ installed.
2. Ollama installed and running.
    - Download from ollama.com.
    - Pull the models you intend to use:

```Bash
ollama pull llama3
ollama pull nomic-embed-text
```

## 🛠️ Installation

1. Clone the repository (or create the folder structure provided).
2. Create a Virtual Environment:
```Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
    
3. Install Dependencies from requirements.txt:
```Bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a .env file in the root directory to customize your setup:
```TOML
# The LLM to use for chat (must be pulled in Ollama)
MODEL_NAME=llama3

# The model to use for embeddings (must be pulled in Ollama)
EMBEDDING_MODEL=nomic-embed-text

# Paths
DATA_PATH=data/source_docs
VECTOR_STORE_PATH=data/vector_store

# RAG Tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🏃 Usage

1. Add Data: Place your .pdf, .txt or .csv files into the data/source_docs/ folder.
2. Run the Agent:
```Bash
python main.py
```
**Chat:** The system will index your documents (first run only) and then let you ask questions about them.

## 🔄 Updating Data

If you add new files to the data/source_docs/ folder, you need to rebuild the vector index. You can do this by modifying main.py temporarily:
```Python
# Change this line in main.py
rag.create_vector_store(force_rebuild=True)
```
Or simply delete the data/vector_store folder and restart the application.