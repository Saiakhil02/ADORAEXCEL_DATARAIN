from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

def create_table_analysis_chain():
    """
    Create an LLMChain for analyzing table data.
    Returns:
        LLMChain: Configured chain for table analysis
    """
    # Define the prompt template
    prompt_template = """You are an expert data analyst. Analyze the following table data and answer the user's question.
    
    Table: {table_name}
    Sheet: {sheet_name}
    
    Table Data (first 5 rows shown, {total_rows} total rows):
    {table_preview}
    
    User Question: {user_question}
    
    Provide a detailed analysis based on the table data:"""
    
    prompt = PromptTemplate(
        input_variables=["table_name", "sheet_name", "table_preview", "total_rows", "user_question"],
        template=prompt_template
    )
    
    # Initialize the language model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create and return the chain
    return LLMChain(llm=llm, prompt=prompt)

def analyze_table_with_llm(chain, table_data, table_name, sheet_name, user_question):
    """
    Analyze table data using the provided LLMChain.
    
    Args:
        chain (LLMChain): Configured LLMChain
        table_data (pd.DataFrame): Table data to analyze
        table_name (str): Name of the table
        sheet_name (str): Name of the sheet
        user_question (str): User's question about the data
        
    Returns:
        str: Generated analysis
    """
    # Prepare table preview (first 5 rows)
    table_preview = table_data.head().to_string()
    
    # Get the analysis from LLM
    response = chain.invoke({
        "table_name": table_name,
        "sheet_name": sheet_name,
        "table_preview": table_preview,
        "total_rows": len(table_data),
        "user_question": user_question
    })
    
    return response['text']

def create_text_analysis_chain():
    """
    Create an LLMChain for analyzing text data from files like TXT and PDF.
    Returns:
        LLMChain: Configured chain for text analysis
    """
    # Define the prompt template for text analysis
    prompt_template = """You are an expert document analyst. Analyze the following text data and answer the user's question.
    
    Document: {file_name}
    File Type: {file_type}
    
    Document Content:
    {content_preview}
    
    User Question: {user_question}
    
    Provide a detailed analysis based on the document content:"""
    
    prompt = PromptTemplate(
        input_variables=["file_name", "file_type", "content_preview", "user_question"],
        template=prompt_template
    )
    
    # Initialize the language model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create and return the chain
    return LLMChain(llm=llm, prompt=prompt)

def analyze_text_with_llm(chain: LLMChain, text_data: Dict[str, Any], user_question: str) -> str:
    """
    Analyze text data using the provided LLMChain.
    
    Args:
        chain (LLMChain): Configured LLMChain for text analysis
        text_data (Dict): Dictionary containing text data and metadata
        user_question (str): User's question about the text
        
    Returns:
        str: Generated analysis
    """
    # Prepare content preview (first 2000 characters)
    content = text_data.get('content', '')
    content_preview = content[:2000] + "..." if len(content) > 2000 else content
    
    # Get the analysis from LLM
    response = chain.invoke({
        "file_name": text_data.get('file_name', 'Document'),
        "file_type": text_data.get('file_type', 'text').upper(),
        "content_preview": content_preview,
        "user_question": user_question
    })
    
    return response['text']
