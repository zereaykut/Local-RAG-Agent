from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import settings

class LocalAgent:
    def __init__(self, retriever):
        self.llm = ChatOllama(model=settings.MODEL_NAME)
        self.retriever = retriever
        self.chain = self._build_chain()

    def _build_chain(self):
        template = """You are a helpful local AI assistant. Use the context below to answer the question.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, user_query: str):
        return self.chain.invoke(user_query)