from src.rag_engine import RAGPipeline
from src.bot_agent import LocalAgent

def main():
    print("Initializing Local AI Agent...")
    
    # 1. Initialize RAG Pipeline
    rag = RAGPipeline()
    
    # Option: Set force_rebuild=True if added new documents
    rag.create_vector_store(force_rebuild=False) 
    
    # 2. Initialize Agent with the retriever
    agent = LocalAgent(retriever=rag.get_retriever())
    
    print("\n✅ System Ready. Type 'exit' to quit.\n")
    
    # 3. Chat Loop
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        try:
            response = agent.chat(query)
            print(f"AI: {response}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()