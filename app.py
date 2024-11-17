from typing import List, Dict, Any
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import logging
import chainlit as cl
import numpy as np
import os
import graphviz
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
LOG_FILE = "logs/chatbot.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   handlers=[logging.FileHandler(LOG_FILE),
                            logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Constants
DOCS_DIR = "docs"
SIMILARITY_THRESHOLD = 0.1

class KnowledgeBase:
    """Knowledge base for the chatbot."""
    def __init__(self, docs_dir: str = DOCS_DIR):
        """Initialize the knowledge base."""
        self.docs_dir = docs_dir
        self.knowledge_base = []
        self.embeddings_cache = None
        logger.info(f"Initializing KnowledgeBase with docs directory: {docs_dir}")
        try:
            self.embeddings = OllamaEmbeddings(
                model="mistral",
                base_url="http://localhost:11434"
            )
            # Test the connection
            self.embeddings.embed_query("test")
            logger.info("Successfully connected to Ollama embeddings service")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {str(e)}")
            raise RuntimeError("Failed to initialize Ollama. Is the service running?")
        
        self._load_knowledge_base()
        if not self.knowledge_base:
            logger.warning("No documents loaded into knowledge base!")
        self.embeddings_cache = self._compute_embeddings()

    def _load_knowledge_base(self) -> None:
        """Load documents from the docs directory."""
        try:
            for filename in os.listdir(self.docs_dir):
                if filename.endswith(".md"):
                    filepath = os.path.join(self.docs_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split content into sections
                    sections = content.split("\n\n")
                    for section in sections:
                        if section.strip():
                            self.knowledge_base.append({
                                'content': section.strip(),
                                'source': filename,
                                'title': self._extract_title(section)
                            })
                    logger.info(f"Successfully loaded {len(sections)} sections from {filename}")
            
            logger.info(f"Total sections in knowledge base: {len(self.knowledge_base)}")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise

    def _extract_title(self, section: str) -> str:
        """Extract a title from a section of text."""
        lines = section.split("\n")
        for line in lines:
            if line.strip().startswith("#"):
                return line.strip("# ")
        return "Section"

    def _compute_embeddings(self) -> np.ndarray:
        """Compute embeddings for all sections in the knowledge base."""
        try:
            if not self.knowledge_base:
                logger.warning("No content to compute embeddings for")
                return np.array([])
            
            embeddings = []
            for item in self.knowledge_base:
                embedding = self.embeddings.embed_query(item['content'])
                # Normalize the embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)
            
            logger.info(f"Successfully computed embeddings for {len(embeddings)} sections")
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            raise

    def get_relevant_content(self, query: str) -> str:
        """Search the knowledge base for content relevant to the query."""
        try:
            if not self.knowledge_base or self.embeddings_cache is None or len(self.embeddings_cache) == 0:
                logger.error("Knowledge base or embeddings not properly initialized")
                return "I apologize, but I'm having trouble accessing my knowledge base."
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            # Calculate similarities
            similarities = np.dot(self.embeddings_cache, query_embedding)
            
            # Get top matches
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score < SIMILARITY_THRESHOLD:
                logger.info(f"No relevant content found for query: {query} (best score: {best_score})")
                return "I couldn't find any directly relevant information in my knowledge base."
            
            best_match = self.knowledge_base[best_idx]
            logger.info(f"Found relevant content from {best_match['source']} with score {best_score}")
            
            return f"From {best_match['source']}:\n{best_match['content']}"
            
        except Exception as e:
            logger.error(f"Error retrieving relevant content: {str(e)}")
            raise

class SnapLabsAssistant:
    """Main assistant class that handles conversation and knowledge retrieval."""
    def __init__(self):
        self.llm = Ollama(
            model="mistral",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        self.kb = KnowledgeBase()
        
        # Create system prompt
        self.system_prompt = """You are a professional and helpful AI assistant for SnapLabs. 
        Your role is to help users understand SnapLabs' services and capabilities.
        
        When responding:
        1. If the user asks about SnapLabs' services or information, use the provided context.
        2. If no context is provided or the question is general, provide a helpful response based on the query.
        3. Always be polite, accurate, and concise.
        4. Format your responses in a clear and readable way.
        
        Context: {context}
        
        User Query: {query}"""

    async def process_message(self, message: str) -> str:
        """Process an incoming message and return a response."""
        try:
            # Get relevant content from knowledge base
            context = self.kb.get_relevant_content(message)
            
            # Create prompt with context
            prompt = ChatPromptTemplate.from_template(self.system_prompt)
            
            # Generate response
            response = self.llm.invoke(
                prompt.format(
                    context=context,
                    query=message
                )
            )
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error while processing your message. Please try again."

# Initialize the assistant
assistant = SnapLabsAssistant()

# FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your website's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(chat_message: ChatMessage):
    try:
        response = await assistant.process_message(chat_message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@cl.on_chat_start
async def start():
    """Initialize a new chat session."""
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Hello! I'm the SnapLabs AI Assistant. How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming chat messages."""
    try:
        # Create a processing action
        action = cl.Action(name="processing", value="true", label="Thinking...")
        msg = cl.Message(content="", actions=[action])
        await msg.send()
        
        # Get response from assistant
        response = await assistant.process_message(message.content)
        
        # Remove processing message and send response
        await msg.remove()
        await cl.Message(content=response).send()
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logger.error(error_msg)
        if 'msg' in locals():
            await msg.remove()
        await cl.Message(content="I apologize, but something went wrong. Please try again.").send()

if __name__ == "__main__":
    # Create architecture diagram
    dot = graphviz.Digraph(comment='SnapLabs AI Assistant Architecture')
    
    # Add nodes
    dot.attr('node', shape='box', style='rounded')
    dot.node('User', 'User')
    dot.node('Chainlit', 'Chainlit Interface')
    dot.node('Assistant', 'SnapLabs Assistant')
    dot.node('LLM', 'Ollama LLM\n(Mistral)')
    dot.node('KB', 'Knowledge Base')
    dot.node('Docs', 'Markdown Docs')
    
    # Add edges
    dot.edge('User', 'Chainlit', 'sends message')
    dot.edge('Chainlit', 'Assistant', 'routes message')
    dot.edge('Assistant', 'KB', 'queries')
    dot.edge('KB', 'Docs', 'reads')
    dot.edge('Assistant', 'LLM', 'generates response')
    dot.edge('LLM', 'Assistant', 'returns response')
    dot.edge('Assistant', 'Chainlit', 'sends response')
    dot.edge('Chainlit', 'User', 'displays response')
    
    # Save the diagram
    dot.render('architecture', format='png', cleanup=True)
    logger.info("Architecture diagram saved as 'architecture.png'")

    # Ensure Ollama service is running
    try:
        assistant.llm.invoke("test")
        logger.info("Successfully connected to Ollama service")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama service: {str(e)}")
        logger.error("Please make sure Ollama is running and the mistral model is installed")
        raise RuntimeError("Failed to connect to Ollama service")
