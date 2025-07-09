from typing import List, Dict, Any, Optional, Tuple, Union
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import logging
import re
import pandas as pd
import json
from visualization import viz_generator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Chatbot:
    """
    Handles the chat functionality for interacting with documents.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        """
        Initialize the chatbot with a specific model and settings.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Controls randomness in the response generation (0-1)
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.system_prompt = self._get_system_prompt()
        self.conversation_chain = self._create_conversation_chain()
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt that sets the behavior of the chatbot.
        
        Returns:
            str: The system prompt
        """
        return """You are AskAdora, an AI Document Assistant. Your task is to help users understand and extract 
        information from their documents. Follow these guidelines:
        
        1. Be comprehensive and thorough in your responses, using all available context from the document chunks.
        2. If the context contains multiple relevant chunks, synthesize information across them to provide a complete answer.
        3. When appropriate, reference specific chunks or sources (e.g., "According to Chunk 3...").
        4. If the information is spread across multiple chunks, combine it into a coherent response.
        5. Be accurate and factual - if you're unsure about something, say so.
        6. For complex queries, break down your response into clear, organized sections.
        7. If the user asks about specific parts of the document, reference the relevant chunks in your response.
        
        The context you receive will be formatted with chunk numbers and metadata. Use this to provide 
        well-sourced answers and help users understand where information is coming from in their documents."""
    
    def _create_conversation_chain(self) -> LLMChain:
        """
        Create a conversation chain with the current LLM and memory.
        
        Returns:
            LLMChain: The configured conversation chain
        """
        # Create prompt templates
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_prompt
        )
        human_template = "{input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        # Create the chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        # Create and return the conversation chain
        return LLMChain(
            llm=self.llm,
            prompt=chat_prompt,
            memory=self.memory,
            verbose=True
        )
    
    def _is_visualization_request(self, user_input: str) -> bool:
        """Check if the user is requesting a visualization."""
        visualization_keywords = [
            'graph', 'chart', 'plot', 'visualization', 'visualize',
            'show me a', 'display a', 'create a', 'generate a',
            'bar', 'line', 'pie', 'histogram', 'scatter', 'boxplot',
            'compare', 'relationship', 'distribution', 'trend'
        ]
        return any(keyword in user_input.lower() for keyword in visualization_keywords)
    
    def _extract_dataframe_info(self, context: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Try to extract a pandas DataFrame from the context or previous messages.
        
        Args:
            context: The context string that might contain DataFrame information
            
        Returns:
            Optional[pd.DataFrame]: Extracted DataFrame or None if not found
        """
        # First try to find a DataFrame in the context
        if context:
            try:
                # Look for tabular data in markdown format
                table_pattern = r'\|(.+)\|\n\|[-:]+\|\n((?:\|.*\|\n)+)'
                matches = re.finditer(table_pattern, context)
                
                for match in matches:
                    try:
                        # Extract headers and rows
                        headers = [h.strip() for h in match.group(1).split('|') if h.strip()]
                        rows = []
                        for row in match.group(2).split('\n'):
                            if not row.strip():
                                continue
                            cells = [c.strip() for c in row.split('|') if c.strip()]
                            if len(cells) == len(headers):
                                rows.append(cells)
                        
                        if headers and rows:
                            return pd.DataFrame(rows[1:], columns=headers)
                    except Exception as e:
                        logger.debug(f"Error parsing markdown table: {e}")
                
                # Look for JSON data
                json_pattern = r'```(?:json\n)?(\{.*?\})\s*```'
                match = re.search(json_pattern, context, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        if isinstance(data, dict):
                            return pd.DataFrame([data])
                        elif isinstance(data, list):
                            return pd.DataFrame(data)
                    except json.JSONDecodeError:
                        pass
                
                # Look for list of dictionaries
                dict_pattern = r'\[\s*\{.*?\}\s*\]'
                match = re.search(dict_pattern, context, re.DOTALL)
                if match:
                    try:
                        data = eval(match.group(0))
                        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                            return pd.DataFrame(data)
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Error extracting DataFrame from context: {e}")
        
        # If no DataFrame found in context, check if we have any tabular data in previous messages
        if hasattr(self, 'memory') and hasattr(self.memory, 'chat_memory'):
            for msg in reversed(self.memory.chat_memory.messages):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    try:
                        # Try to parse as JSON
                        data = json.loads(msg.content)
                        if isinstance(data, dict):
                            return pd.DataFrame([data])
                        elif isinstance(data, list):
                            return pd.DataFrame(data)
                    except json.JSONDecodeError:
                        # Not JSON, try other formats
                        pass
        
        return None
    
    def get_response(self, user_input: str, context: Optional[str] = None) -> Tuple[str, Optional[Dict]]:
        """
        Get a response from the chatbot based on user input and optional context.
        
        Args:
            user_input: The user's message
            context: Optional context from document search
            
        Returns:
            Tuple containing:
                - str: The chatbot's response
                - Optional[Dict]: Visualization data if applicable
        """
        visualization_data = None
        
        # Check if this is a visualization request
        if self._is_visualization_request(user_input):
            df = self._extract_dataframe_info(context)
            
            if df is not None and not df.empty:
                try:
                    # Generate the visualization
                    visualization_data = viz_generator.create_visualization(df, user_input)
                    
                    # Add a helpful response
                    response = "I've created a visualization for you:"
                    return response, visualization_data
                    
                except Exception as e:
                    logger.error(f"Error generating visualization: {e}")
                    response = f"I had trouble creating that visualization: {str(e)}. Could you try being more specific about what you'd like to see?"
                    return response, None
            else:
                # If no data found for visualization, ask for clarification
                response = "I couldn't find any data to visualize. Could you provide the data in a table format or describe what you'd like to see?"
                return response, None
        
        # If not a visualization request or visualization failed, use the LLM
        # Add context to the user input if provided
        if context:
            user_input = f"Context: {context}\n\nQuestion: {user_input}"
        
        # Get response from the conversation chain
        response = self.conversation_chain.run(input=user_input)
        
        # After getting the response, check if it contains tabular data that could be visualized
        if not visualization_data and self._is_visualization_request(user_input):
            df = self._extract_dataframe_info(response)
            if df is not None and not df.empty:
                try:
                    visualization_data = viz_generator.create_visualization(df, user_input)
                    response = f"{response}\n\nI've also created a visualization of the data:"
                except Exception as e:
                    logger.debug(f"Could not create visualization from response: {e}")
        
        return response, visualization_data
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.memory.clear()
        self.conversation_chain = self._create_conversation_chain()

# Global instance
chatbot = Chatbot()
