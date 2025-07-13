import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import tempfile
import time
import json
from pathlib import Path
import openai
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
from database import init_db, get_db
from file_processor import FileProcessor
from vector_store import vector_store_manager
from chatbot import Chatbot
from models import Document, ChatMessage, DocumentChunk
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌ OpenAI API key not found in environment variables")
    st.stop()
client = openai.OpenAI(api_key=openai.api_key)

# Initialize database
try:
    if init_db():
        st.success("✅ Database initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to initialize database: {e}")
    st.stop()

# Initialize file processor
try:
    file_processor = FileProcessor()
except Exception as e:
    st.error(f"❌ Failed to initialize file processor: {e}")
    st.stop()

# Initialize chatbot in session state if not exists
if 'chatbot' not in st.session_state:
    try:
        st.session_state.chatbot = Chatbot()
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {e}")
        st.stop()

# Set page configuration
st.set_page_config(
    page_title="AskAdora - Document Analyzer with AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .stTextInput>div>div>input { font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {}
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

# Function to generate Plotly code from user query
def get_plotly_code_from_input(input_text, query):
    prompt = f"""
You are a Python coding assistant.
Your job is to read the following tabular data and generate full Python code using pandas and plotly to create a chart based on the user's query. The query may specify a chart type (e.g., histogram, scatter, bar) and one or two columns. Use the exact data provided without any modifications, additions, or deletions. Ensure all arrays in the output code have the same length as the input data. For two-column charts, use the first column as the x-axis and the second as the y-axis. For single-column charts (e.g., histogram), use the specified column. Do not use generic terms like 'Axis A' or 'Axis B'. Ensure all numerical values are native Python types (float or int), not NumPy types like np.float64. For bar or histogram charts with categorical data, aggregate appropriately (e.g., counts).
Interpret the query flexibly to extract the chart type and column(s), supporting varied formats (e.g., 'generate a histogram for ZIP', 'plot ZIP vs Year Built as scatter', 'make a bar graph with ZIP and Branch No').
Only output valid Python code. Do not explain anything. Assume any necessary imports.
Input Data:
{input_text}
Query: {query}
Output:
(Python code only)
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = response.choices[0].message.content
        if code.startswith("```python"):  
            code = code.replace("```python", "").replace("```", "").strip()  
        elif code.startswith("```"):
            code = code.replace("```", "").strip()  
        return code  
    except Exception as e:  
        print(f"OpenAI API error: {str(e)}")  
        return None  

# Function to format document chunk info for display  
def _format_chunk_info(chunk, chunk_number):  
    try:  
        if hasattr(chunk, 'chunk_text'):  
            chunk_text = chunk.chunk_text  
            metadata = chunk.chunk_metadata or {}  
            chunk_idx = chunk.chunk_index  
        else:  
            chunk_text = getattr(chunk, 'page_content', '')  
            metadata = getattr(chunk, 'metadata', {}) or {}  
            chunk_idx = metadata.get('chunk_index', chunk_number)  
        chunk_info = f"[Chunk {chunk_number}]"  
        if 'source' in metadata:  
            chunk_info += f" (Source: {metadata['source']})"  
        if 'page' in metadata:  
            chunk_info += f" (Page: {metadata['page']})"  
        if 'chunk_index' in metadata:  
            chunk_info += f" (Index: {metadata['chunk_index']})"  
        chunk_info += f"\n{chunk_text}"  
        return chunk_info  
    except Exception as e:  
        print(f"Error formatting chunk {chunk_number}: {str(e)}")  
        return f"[Chunk {chunk_number} - Error]"  

# Function to get or create chat session for current document  
def get_current_chat_session():  
    if st.session_state.current_document is None:  
        return []  
    if st.session_state.current_document not in st.session_state.chat_sessions:  
        st.session_state.chat_sessions[st.session_state.current_document] = []  
    return st.session_state.chat_sessions[st.session_state.current_document]  

# Page title and description  
st.title("📚 AskAdora - Document Analyzer with AI")  
st.markdown("Upload your documents and ask questions about their content. Supports PDF, DOCX, XLSX, XLS, and TXT files.")  

# Sidebar for file upload and settings  
with st.sidebar:  
    st.header("📂 Document Management")  
    
    # Initialize file_uploader_key in session state if it doesn't exist
    if 'file_uploader_key' not in st.session_state:
        st.session_state['file_uploader_key'] = 0
        
    # File uploader with unique key that changes when files are removed
    uploaded_files = st.file_uploader(  
        "Upload documents (PDF, DOCX, XLSX, XLS, TXT)",  
        type=["pdf", "docx", "xlsx", "xls", "txt"],  
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    # Process each uploaded file  
    if uploaded_files:  
        # Check for duplicate files
        duplicate_files = [file.name for file in uploaded_files 
                         if file.name in st.session_state.get('documents', {})]
        
        if duplicate_files:
            st.error(f"The following files already exist and were not uploaded: {', '.join(duplicate_files)}")
        
        # Only process new files that aren't already in documents
        files_to_process = [file for file in uploaded_files 
                          if file.name not in st.session_state.get('documents', {})]
                          
        for uploaded_file in files_to_process:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file temporarily
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name
                    temp_file.close()
                    
                    # Store temp file path for later cleanup
                    temp_files = st.session_state.get('temp_files', {})
                    temp_files[uploaded_file.name] = temp_file_path
                    st.session_state['temp_files'] = temp_files

                    file_content = ""
                    doc_chunks = []

                    # Handle TXT files
                    if uploaded_file.type == 'text/plain':
                        file_content = uploaded_file.read().decode('utf-8')
                        doc_chunks.append({
                            'chunk_text': file_content,
                            'chunk_index': 0,
                            'metadata': {"source": uploaded_file.name, "chunk_type": "full_text"}
                        })
                        st.session_state['documents'][uploaded_file.name] = {
                            'id': None,
                            'name': uploaded_file.name,
                            'size': f"{len(uploaded_file.getvalue()) / 1024:.1f} KB",
                            'type': uploaded_file.type or 'Unknown',
                            'uploaded_at': None,
                            'processed': True,
                            'content': file_content,
                            'chunk_count': len(doc_chunks),
                            'dataframe': None,
                            'temp_file_path': temp_file_path
                        }
                    elif uploaded_file.type in [
                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        'application/vnd.ms-excel'
                    ]:
                        try:
                            df = pd.read_excel(uploaded_file, nrows=200)
                            file_content = df.to_string()
                            st.session_state['documents'][uploaded_file.name] = {
                                'id': None,
                                'name': uploaded_file.name,
                                'size': f"{len(uploaded_file.read()) / 1024:.1f} KB",
                                'type': uploaded_file.type or 'Unknown',
                                'uploaded_at': None,
                                'processed': True,
                                'content': file_content,
                                'chunk_count': 0,
                                'dataframe': df
                            }
                            for i, row in df.iterrows():
                                chunk_text = f"Row {i+1}:\n"
                                for col in df.columns:
                                    chunk_text += f"{col}: {row[col]}\n"
                                doc_chunks.append({
                                    'chunk_text': chunk_text,
                                    'chunk_index': i,
                                    'metadata': {"source": uploaded_file.name, "row": i, "chunk_type": "data_row"}
                                })
                            st.session_state['documents'][uploaded_file.name]['chunk_count'] = len(doc_chunks)
                        except Exception as e:
                            st.warning(f"Could not process Excel content: {str(e)}")
                            continue
                    elif uploaded_file.type == 'application/pdf':
                        # Process PDF with FileProcessor
                        try:
                            uploaded_file.seek(0)
                            processor = FileProcessor()
                            result = processor.process_file(uploaded_file, original_filename=uploaded_file.name)
                            if not result:
                                raise ValueError("Failed to process PDF")
                            file_content = result.get('text', '').strip()
                            if file_content:
                                chunks = processor.text_splitter.split_text(file_content)
                                for i, chunk in enumerate(chunks):
                                    doc_chunks.append({
                                        'chunk_text': chunk,
                                        'chunk_index': i,
                                        'metadata': {"source": uploaded_file.name, "chunk_type": "text"}
                                    })
                                st.success(f"Extracted {len(chunks)} chunks from PDF")
                            else:
                                raise ValueError("No text extracted from PDF")
                            # Save doc info
                            st.session_state['documents'][uploaded_file.name] = {
                                'id': None,
                                'name': uploaded_file.name,
                                'size': f"{len(uploaded_file.read()) / 1024:.1f} KB",
                                'type': uploaded_file.type or 'application/pdf',
                                'uploaded_at': None,
                                'processed': True,
                                'content': file_content,
                                'chunk_count': len(doc_chunks),
                                'dataframe': None
                            }
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")
                            continue
                    # Save file and chunks to database
                    db = next(get_db())
                    try:
                        # Persist document
                        existing_doc = db.query(Document).filter(Document.filename == uploaded_file.name).first()
                        if existing_doc:
                            doc = existing_doc
                            doc.file_size = len(uploaded_file.read())
                            doc.processed = True
                            db.add(doc)
                        else:
                            doc = Document(
                                filename=uploaded_file.name,
                                file_type=uploaded_file.type or 'application/octet-stream',
                                file_size=len(uploaded_file.read()),
                                processed=True
                            )
                            db.add(doc)
                        db.commit()
                        db.refresh(doc)
                        # Save chunks
                        for chunk in doc_chunks:
                            db_chunk = DocumentChunk(
                                document_id=doc.id,
                                chunk_text=chunk['chunk_text'],
                                chunk_index=chunk['chunk_index'],
                                chunk_metadata=chunk['metadata']
                            )
                            db.add(db_chunk)
                        db.commit()

                        # Convert chunks to LangChain documents
                        from langchain.docstore.document import Document as LCDocument
                        lc_docs = [
                            LCDocument(
                                page_content=chunk['chunk_text'],
                                metadata=chunk['metadata']
                            )
                            for chunk in doc_chunks
                        ]
                        # Add to vector store
                        if lc_docs:
                            collection_name = os.path.splitext(uploaded_file.name)[0]
                            organization_id = 1
                            excel_file_id = doc.id
                            vector_store_manager.add_documents(
                                lc_docs,
                                namespace=collection_name,
                                organization_id=organization_id,
                                excel_file_id=excel_file_id
                            )
                        # Update session
                        st.session_state['chat_sessions'][uploaded_file.name] = []
                        if st.session_state['current_document'] is None:
                            st.session_state['current_document'] = uploaded_file.name
                        st.success(f"Successfully processed {uploaded_file.name}")
                        
                        # Clean up the temporary file after successful processing
                        if 'temp_files' in st.session_state and uploaded_file.name in st.session_state['temp_files']:
                            try:
                                temp_path = st.session_state['temp_files'][uploaded_file.name]
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                                del st.session_state['temp_files'][uploaded_file.name]
                            except Exception as cleanup_error:
                                logger.error(f"Error during cleanup: {cleanup_error}")
                        
                        # Increment the file uploader key to reset the file uploader
                        st.session_state['file_uploader_key'] += 1
                        st.rerun()
                            
                    except Exception as e:
                        db.rollback()
                        st.error(f"Error saving document: {str(e)}")
                    finally:
                        db.close()
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    # Clean up any temporary files
                    if 'temp_files' in st.session_state and uploaded_file.name in st.session_state['temp_files']:
                        try:
                            temp_path = st.session_state['temp_files'][uploaded_file.name]
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            del st.session_state['temp_files'][uploaded_file.name]
                        except Exception as cleanup_error:
                            logger.error(f"Error during cleanup: {cleanup_error}")
                    # Remove from documents if it was added
                    if uploaded_file.name in st.session_state.get('documents', {}):
                        del st.session_state['documents'][uploaded_file.name]

    # Document list display
    if st.session_state['documents']:
        st.subheader("📋 Uploaded Documents")
        files_to_remove = []
        
        for doc_name, doc in list(st.session_state['documents'].items()):
            is_active = st.session_state['current_document'] == doc_name
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"📄 {doc_name}" + (" (Active)" if is_active else ""),
                    key=f"select_{doc_name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state['current_document'] = doc_name
                    st.rerun()
            with col2:
                if st.button("❌", key=f"remove_{doc_name}"):
                    # Clean up any associated temporary files
                    if 'temp_files' in st.session_state and doc_name in st.session_state['temp_files']:
                        try:
                            temp_path = st.session_state['temp_files'][doc_name]
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            del st.session_state['temp_files'][doc_name]
                        except Exception as cleanup_error:
                            logger.error(f"Error during cleanup: {cleanup_error}")
                    files_to_remove.append(doc_name)
        
        # Process file removals after the loop to avoid modifying the dict during iteration
        if files_to_remove:
            with st.spinner("Removing files..."):
                for doc_name in files_to_remove:
                    # Remove from session state
                    if doc_name in st.session_state['documents']:
                        del st.session_state['documents'][doc_name]
                    if doc_name in st.session_state['chat_sessions']:
                        del st.session_state['chat_sessions'][doc_name]
                    
                    # Update current document if needed
                    if st.session_state.get('current_document') == doc_name:
                        st.session_state['current_document'] = next(iter(st.session_state['documents']), None)
                    
                    # Remove from database
                    db = next(get_db())
                    try:
                        doc = db.query(Document).filter(Document.filename == doc_name).first()
                        if doc:
                            db.query(ChatMessage).filter(ChatMessage.document_id == doc.id).delete()
                            db.delete(doc)
                            db.commit()
                    except Exception as e:
                        db.rollback()
                        st.error(f"Error deleting document: {str(e)}")
                    finally:
                        db.close()
                
                # Clear the file uploader state by incrementing the key
                st.session_state['file_uploader_key'] += 1
                st.rerun()
            st.caption(f"Type: {doc['type']} | Size: {doc['size']} | Uploaded: {doc.get('uploaded_at', 'Unknown')}")
            st.markdown("---")  
    
    # Settings  
    st.subheader("⚙️ Settings")  
    model_name = st.selectbox(  
        "AI Model",  
        ["gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],  
        index=0,  
        help="Select the AI model for responses"  
    )  
    temperature = st.slider(  
        "Creativity",  
        min_value=0.0,  
        max_value=1.0,  
        value=0.7,  
        step=0.1,  
        help="Higher makes responses more random"  
    )  

# Chat interface  
if st.session_state['current_document']:  
    st.header(f"💬 Chat with {st.session_state['current_document']}")  
    current_chat = get_current_chat_session()  

    for message in current_chat:  
        with st.chat_message(message["role"]):  
            st.write(message["content"])  
            if "visualization" in message and message["visualization"]:  
                vis_data = message["visualization"]  
                if vis_data.get('type') == 'plotly':  
                    try:  
                        fig = go.Figure(vis_data['data'])  
                        st.plotly_chart(fig, use_container_width=True)  
                    except Exception as e:  
                        st.error(f"Error displaying visualization: {str(e)}")  
                else:  
                    st.json(vis_data)  

    if prompt := st.chat_input(f"Ask about {st.session_state['current_document']}..."):  
        current_chat.append({"role": "user", "content": prompt, "visualization": None})  
        db = next(get_db())  
        try:  
            doc = db.query(Document).filter(Document.filename == st.session_state['current_document']).first()  
            if not doc:  
                st.error("Document not found in database. Please re-upload the document.")  
                db.close()  
                st.rerun()

            # Save the database document ID before we overwrite the doc variable
            document_id = doc.id

            msg = ChatMessage(  
                document_id=document_id,  
                role="user",  
                content=prompt,  
                message_metadata=None  
            )  
            db.add(msg)  
            db.commit()  

            with st.spinner("Analyzing document..."):  
                context = None  
                visualization = None  
                response = "Processing your request..."  
                try:  
                    if st.session_state['current_document'] in st.session_state['documents']:  
                        doc_name = os.path.splitext(st.session_state['current_document'])[0]  
                        # Use a different variable name to avoid shadowing the database document
                        doc_dict = st.session_state['documents'][st.session_state['current_document']]  
                        # Check for graph-related query  
                        graph_keywords = ['plot', 'graph', 'chart', 'histogram', 'scatter', 'bar']  
                        is_graph_query = any(k in prompt.lower() for k in graph_keywords)  
                        if is_graph_query and 'dataframe' in doc_dict and doc_dict['dataframe'] is not None:  
                            df = doc_dict['dataframe']  
                            available_columns = df.columns.tolist()  
                            mentioned_columns = [col for col in available_columns if col.lower() in prompt.lower()]  

                            if mentioned_columns:  
                                rows = []  
                                original_data = {col: [] for col in mentioned_columns}  
                                for index, row in df[mentioned_columns].iterrows():  
                                    row_values = [  
                                        float(row[col]) if isinstance(row[col], (np.floating, np.integer)) else row[col]  
                                        if not pd.isna(row[col]) else None for col in mentioned_columns  
                                    ]  
                                    rows.append(str(row_values).replace("None", "null"))  
                                    for col, value in zip(mentioned_columns, row_values):  
                                        original_data[col].append(value)  

                                input_text = f"data = [\n"  
                                for row_val in rows:  
                                    input_text += f"    {row_val},\n"  
                                input_text += "]"  

                                plotly_code = get_plotly_code_from_input(input_text, prompt)  
                                if plotly_code:  
                                    try:  
                                        local_vars = {}  
                                        exec(plotly_code, {}, local_vars)  
                                        if 'fig' in local_vars:  
                                            visualization = {'type': 'plotly', 'data': local_vars['fig'].to_dict()}  
                                            response = f"Generated a graph based on your query: {prompt}"  
                                        else:  
                                            response = "No figure generated in Plotly code."  
                                    except Exception as e:  
                                        response = f"Failed to generate graph: {str(e)}"  
                                else:  
                                    response = "Failed to generate graph due to API error."  
                            else:  
                                response = f"No valid columns found in the query. Available columns: {', '.join(available_columns)}"  
                        else:  
                            # Retrieve relevant chunks from vector store  
                            try:
                                # Get the document ID from the database
                                doc = db.query(Document).filter(Document.filename == st.session_state['current_document']).first()
                                if not doc:
                                    raise ValueError("Document not found in database")
                                    
                                total_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()  
                                k = min(10, total_chunks) if total_chunks else 10  
                                relevant_chunks = vector_store_manager.similarity_search(  
                                    query=prompt,  
                                    k=k,  
                                    namespace=doc_name,  
                                    organization_id=1,  
                                    excel_file_id=doc.id  
                                )  
                                context_chunks = []  
                                used_chunk_indices = set()  
                                for chunk in relevant_chunks:  
                                    chunk_idx = getattr(chunk, "metadata", {}).get('chunk_index', None)  
                                    if chunk_idx is not None:  
                                        used_chunk_indices.add(chunk_idx)  
                                    chunk_info = _format_chunk_info(chunk, len(context_chunks) + 1)  
                                    if chunk_info:  
                                        context_chunks.append(chunk_info)  

                                if len(context_chunks) < 5 and total_chunks and total_chunks > len(context_chunks):  
                                    try:  
                                        adjacent_chunks = []  
                                        for idx in list(used_chunk_indices):  
                                            for offset in [-1, 1]:  
                                                adj_idx = idx + offset  
                                                if 0 <= adj_idx < total_chunks and adj_idx not in used_chunk_indices:  
                                                    adj_chunk = db.query(DocumentChunk).filter(  
                                                        DocumentChunk.document_id == doc.id,  
                                                        DocumentChunk.chunk_index == adj_idx  
                                                    ).first()  
                                                    if adj_chunk:  
                                                        adjacent_chunks.append(adj_chunk)  
                                                        used_chunk_indices.add(adj_idx)  
                                                        if len(adjacent_chunks) >= (5 - len(context_chunks)):  
                                                            break  
                                        for chunk in adjacent_chunks:  
                                            chunk_info = _format_chunk_info(chunk, len(context_chunks) + 1)  
                                            if chunk_info:  
                                                context_chunks.append(chunk_info)  
                                    except Exception as e:  
                                        print(f"DEBUG: Error getting adjacent chunks: {str(e)}")  
                                context = "\n\n".join(context_chunks)  
                                if hasattr(doc_dict, 'metadata') and doc_dict.get('metadata'):  
                                    context = f"Document Metadata: {json.dumps(doc_dict['metadata'], indent=2)}\n\n{context}"  
                            except Exception as e:  
                                print(f"DEBUG: Error retrieving context: {str(e)}")  
                            if not context and 'content' in doc_dict and doc_dict['content']:  
                                context = doc_dict['content']  
                            if not context:  
                                context = "The document appears to be empty or could not be processed. Please re-upload."  
                            # Get response from chatbot  
                            response, _ = st.session_state['chatbot'].get_response(user_input=prompt, context=context)  
                    # Ensure response is string  
                    if not isinstance(response, str):  
                        raise ValueError(f"Invalid response type: {type(response)}")  
                    current_chat.append({"role": "assistant", "content": response, "visualization": visualization})  
                    # Save message to database  
                    ai_msg = ChatMessage(  
                        document_id=doc.id,  
                        role="assistant",  
                        content=response,  
                        message_metadata={"visualization": visualization} if visualization else None  
                    )  
                    db.add(ai_msg)  
                    db.commit()  
                except Exception as e:  
                    st.error(f"Error during response generation: {str(e)}")  
                    current_chat.append({  
                        "role": "assistant",  
                        "content": f"Sorry, I encountered an error: {str(e)}",  
                        "visualization": None  
                    })  
        finally:  
            if 'db' in locals():  
                db.close()  
        st.rerun()  
    # Clear chat history button  
    if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True, key="clear_chat_button"):  
        if st.session_state['current_document']:  
            db = next(get_db())  
            try:  
                # Delete chat messages from database  
                doc = db.query(Document).filter(Document.filename == st.session_state['current_document']).first()  
                if doc:  
                    db.query(ChatMessage).filter(ChatMessage.document_id == doc.id).delete()  
                    db.commit()  
                # Clear session chat  
                if st.session_state['current_document'] in st.session_state['chat_sessions']:  
                    st.session_state['chat_sessions'][st.session_state['current_document']] = []  
                st.success("Chat history cleared.")  
            except Exception as e:  
                db.rollback()  
                st.error(f"Error clearing chat: {str(e)}")  
            finally:  
                db.close()  
            st.rerun()  
else:  
    st.info("📁 Upload and select a document to start chatting.")  

# Tips  
st.sidebar.markdown("---")  
st.sidebar.info(  
    "💡 **Tips:**\n\n"  
    "• Upload multiple documents\n\n"  
    "• Ask in natural language\n\n"  
    "• For Excel files, request graphs (e.g., 'generate a scatter plot of Column1 vs Column2' or 'create a histogram for Column1')\n\n"
    "• The AI will use the content of your documents to answer\n\n"
    "• Try different questions to explore your documents"
)
