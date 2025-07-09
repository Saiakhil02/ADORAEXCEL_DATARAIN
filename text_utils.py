import PyPDF2
from typing import Union, Optional
import os

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        str: Content of the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error reading text file: {str(e)}")

def extract_text(file_path: str) -> dict:
    """Extract text from a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        dict: Dictionary containing the extracted text and metadata
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        content = extract_text_from_pdf(file_path)
        file_type = 'pdf'
    elif file_extension == '.txt':
        content = extract_text_from_txt(file_path)
        file_type = 'text'
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    return {
        'content': content,
        'file_type': file_type,
        'file_name': os.path.basename(file_path),
        'file_extension': file_extension
    }
