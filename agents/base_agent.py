from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferMemory
import logging

logger = logging.getLogger(__name__)

def create_agent(system_prompt: str):
    # Initialize Ollama with Mistral
    llm = Ollama(model="mistral")
    logger.info("Initialized Ollama with Mistral model")
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    logger.info("Created chat prompt template")
    
    # Create the chain
    chain = prompt | llm
    
    async def process_message(message: str) -> str:
        try:
            logger.info(f"Processing message: {message[:100]}...")
            response = chain.invoke({"input": message})
            logger.info(f"Got response: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}")
            raise
    
    return process_message
