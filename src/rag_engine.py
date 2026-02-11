import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import settings

class RAGPipeline:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vector_store = None

    def load_documents(self):
        """Loads PDFs, Text files, and CSVs from the data directory."""
        if not os.path.exists(settings.DATA_PATH):
            os.makedirs(settings.DATA_PATH)
            print(f"Created data directory at {settings.DATA_PATH}. Please add files.")
            return []

        # Loaders for common file types, including CSV
        loaders = {
            ".txt": DirectoryLoader(settings.DATA_PATH, glob="**/*.txt", loader_cls=TextLoader),
            ".pdf": DirectoryLoader(settings.DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".csv": DirectoryLoader(settings.DATA_PATH, glob="**/*.csv", loader_cls=CSVLoader),
        }
        
        docs = []
        for ext, loader in loaders.items():
            try:
                print(f"Checking for {ext} files...")
                loaded_docs = loader.load()
                if loaded_docs:
                    print(f"Successfully loaded {len(loaded_docs)} documents from {ext} files.")
                    docs.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading {ext} files: {e}")
                
        return docs

    def create_vector_store(self, force_rebuild=False):
        """Creates or loads the Chroma vector store."""
        
        # If persistence directory exists and we aren't forcing rebuild, load it
        if os.path.exists(settings.VECTOR_STORE_PATH) and not force_rebuild:
            print("Loading existing Vector Store...")
            self.vector_store = Chroma(
                persist_directory=settings.VECTOR_STORE_PATH, 
                embedding_function=self.embeddings
            )
            return self.vector_store

        print("Creating new Vector Store...")
        documents = self.load_documents()
        if not documents:
            raise ValueError("No documents found to index. Please check your data folder.")

        # For CSVs, chunk size can often be smaller since each row is loaded as a separate document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, 
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Splitting into {len(chunks)} chunks and generating embeddings (this may take a moment)...")
        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings,
            persist_directory=settings.VECTOR_STORE_PATH
        )
        return self.vector_store

    def get_retriever(self):
        if not self.vector_store:
            self.create_vector_store()
        # k=5 means it will retrieve the top 5 most relevant review chunks for every question
        return self.vector_store.as_retriever(search_kwargs={"k": 5})