from typing import List, Dict
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os

class RAGAgent:
    def __init__(self, knowledge_base_path: str):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = Ollama(model="mistral")
        self.knowledge_base_path = knowledge_base_path
        self.vector_store = None
        
    def initialize_vector_store(self, documents: List[Document]):
        """Initialize or load the vector store."""
        if os.path.exists(self.knowledge_base_path):
            self.vector_store = Chroma(
                persist_directory=self.knowledge_base_path,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.knowledge_base_path
            )
    
    def create_rag_chain(self):
        """Create the RAG chain for question answering."""
        # RAG prompt
        template = """You are a helpful AI assistant for Snap Labs and Snap Analytics. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer in a helpful and professional tone:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # RAG chain
        chain = (
            {
                "context": lambda x: self.vector_store.similarity_search(x["question"]),
                "question": lambda x: x["question"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store."""
        if self.vector_store is None:
            self.initialize_vector_store(documents)
        else:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        if self.vector_store is None:
            return "The knowledge base has not been initialized yet."
        
        chain = self.create_rag_chain()
        response = chain.invoke({"question": question})
        return response
