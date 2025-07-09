import io
import os
import tempfile
import logging
import traceback
import PyPDF2
import pandas as pd
import tabula
from typing import Dict, Any, Optional, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_store_manager import VectorStoreManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileProcessor:
    """
    Handles file uploads, extracts text and tables from various file formats,
    and manages vector store integration.
    """
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.csv': self._process_csv,
            '.txt': self._process_text
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = VectorStoreManager()
        
    def _create_documents(self, content: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Split text into documents with metadata.
        
        Args:
            content: Text content to split
            metadata: Optional metadata to add to documents
            
        Returns:
            List of Document objects
        """
        if not metadata:
            metadata = {}
            
        docs = self.text_splitter.create_documents(
            [content],
            [metadata]
        )
        return docs

    def process_file(self, file, original_filename=None):
        """
        Process the uploaded file, extract content, and update vector store.
        
        Args:
            file: Either a file path (str) or file-like object from Streamlit
            original_filename: Original filename (required if file is a path)
            
        Returns:
            Dict containing extracted text and tables, or None if processing fails
        """
        try:
            # Handle both file path and file-like object
            if isinstance(file, str):  # File path
                if not os.path.exists(file):
                    raise FileNotFoundError(f"File not found: {file}")
                file_extension = os.path.splitext(file)[1].lower()
                filename = original_filename or os.path.basename(file)
                
                # Open the file if it's a path
                with open(file, 'rb') as f:
                    file_obj = io.BytesIO(f.read())
                file_obj.name = filename  # Set name for processing
            else:  # File-like object
                file_extension = os.path.splitext(file.name)[1].lower()
                file_obj = file
                filename = file.name
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Process the file
            result = self.supported_formats[file_extension](file_obj)
            if not result:
                return None
                
            # Update vector store with extracted content
            # In a real app, you'd pass the actual organization_id and file_id
            self._update_vector_store(result, filename, organization_id=1, excel_file_id=1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def _update_vector_store(self, file_content: Dict[str, Any], file_name: str, organization_id: int = 1, excel_file_id: int = 1) -> bool:
        """
        Update vector store with processed file content using PineconeService.
        
        Args:
            file_content: Dictionary containing 'text', 'sheets', and 'tables' keys
            file_name: Name of the file being processed
            organization_id: Organization ID for filtering (default: 1)
            excel_file_id: File ID for tracking (default: 1)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            documents = []
            
            # Add main text content
            text_content = file_content.get('text')
            if text_content and isinstance(text_content, str) and text_content.strip():
                metadata = {
                    'source': file_name,
                    'content_type': 'text',
                    'page': 1,
                    'file_type': 'excel'
                }
                # Split text into smaller chunks for better retrieval
                docs = self._create_documents(text_content, metadata)
                documents.extend(docs)
            
            # Add tables with sheet context
            tables = file_content.get('tables', {})
            if isinstance(tables, dict):
                for table_key, table_data in tables.items():
                    try:
                        if not table_data or not isinstance(table_data, dict):
                            continue
                            
                        # Extract table data and metadata
                        table_rows = table_data.get('data', [])
                        if not table_rows:
                            continue
                            
                        # Convert to DataFrame for consistent processing
                        df = pd.DataFrame(table_rows)
                        if df.empty:
                            continue
                            
                        # Get sheet and table names
                        sheet_name = table_data.get('sheet', 'Unknown')
                        table_name = table_data.get('table_name', table_key)
                        
                        # Create a more structured text representation
                        table_text = self._create_table_text_representation(table_data, max_rows=20)
                        
                        # Create metadata
                        metadata = {
                            'source': file_name,
                            'content_type': 'table',
                            'file_type': 'excel',
                            'sheet': sheet_name,
                            'table_name': table_name,
                            'full_path': f"{file_name} › {sheet_name} › {table_name}",
                            'rows': len(df),
                            'columns': ', '.join([str(c) for c in df.columns.tolist()])
                        }
                        
                        # Create and add documents - one document per table
                        docs = self._create_documents(table_text, metadata)
                        documents.extend(docs)
                        
                        # Also add individual rows as separate documents for better retrieval
                        for idx, row in df.head(50).iterrows():  # Limit to first 50 rows
                            row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                            row_metadata = metadata.copy()
                            row_metadata.update({
                                'row_index': idx,
                                'content_type': 'table_row',
                                'full_path': f"{file_name} › {sheet_name} › {table_name} › Row {idx+1}"
                            })
                            row_doc = self._create_documents(
                                f"Sheet: {sheet_name}\nTable: {table_name}\nRow {idx+1}:\n{row_text}",
                                row_metadata
                            )
                            documents.extend(row_doc)
                        
                    except Exception as table_error:
                        logger.error(f"Error processing table '{table_key}': {str(table_error)}")
                        logger.error(traceback.format_exc())
                        continue
            
            # Update or create vector store and save chunks to database
            if documents:
                store_name = os.path.splitext(os.path.basename(file_name))[0]
                store_name = ''.join(c if c.isalnum() else '_' for c in store_name)
                
                # Get the document from database
                from database import get_db
                from models import Document, DocumentChunk
                
                db = next(get_db())
                try:
                    # Find the document in the database
                    doc = db.query(Document).filter(Document.filename == file_name).first()
                    if not doc:
                        logger.error(f"Document {file_name} not found in database")
                        return False
                    
                    # Delete existing chunks for this document
                    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
                    
                    # Save new chunks
                    for i, doc_chunk in enumerate(documents):
                        chunk = DocumentChunk(
                            document_id=doc.id,
                            chunk_index=i,
                            chunk_text=doc_chunk.page_content,
                            chunk_metadata=doc_chunk.metadata
                        )
                        db.add(chunk)
                    
                    db.commit()
                    logger.info(f"Saved {len(documents)} chunks for document {file_name}")
                    
                    # Update vector store using new PineconeService interface
                    namespace = f"{file_name}_{store_name}"
                    success = self.vector_store.add_documents(
                        documents, 
                        namespace=namespace,
                        organization_id=organization_id,
                        excel_file_id=excel_file_id
                    )
                    
                    if success:
                        logger.info(f"Successfully updated vector store for {file_name}")
                        return True
                    else:
                        logger.error(f"Failed to update vector store for {file_name}")
                        return False
                        
                except Exception as e:
                    db.rollback()
                    logger.error(f"Error saving document chunks to database: {str(e)}")
                    logger.error(traceback.format_exc())
                    return False
                finally:
                    db.close()
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _process_pdf(self, file) -> Dict[str, Any]:
        """Extract text and tables from PDF files with enhanced processing."""
        result = {"text": "", "tables": {}, "processing_errors": []}
        temp_file = None
        
        try:
            # Convert file to bytes if it's a file-like object
            if hasattr(file, 'read'):
                file.seek(0)  # Ensure we're at the start of the file
                file_content = file.read()
                
                # Check if file is empty
                if not file_content:
                    raise ValueError("Uploaded PDF file is empty")
                    
                # Check if it's actually a PDF by looking for the PDF header
                if file_content[:4] != b'%PDF':
                    raise ValueError("The uploaded file does not appear to be a valid PDF")
                
                # Create a temporary file for processing
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(file_content)
                temp_file.flush()
                file_path = temp_file.name
            else:
                file_path = file
            
            # Try PyPDF2 first
            try:
                with open(file_path, 'rb') as f:
                    try:
                        logger.info(f"Attempting to read PDF file: {file_path}")
                        pdf_reader = PyPDF2.PdfReader(f)
                        if not pdf_reader.pages:
                            raise ValueError("PDF contains no pages")
                            
                        logger.info(f"Processing PDF with {len(pdf_reader.pages)} pages")
                        
                        # Extract text from all pages
                        text = ""
                        for i, page in enumerate(pdf_reader.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    text += f"{page_text}\n\n"
                                else:
                                    logger.warning(f"Page {i+1} contains no extractable text")
                            except Exception as page_error:
                                logger.warning(f"Could not extract text from page {i+1}: {str(page_error)}")
                                continue
                        
                        if text.strip():
                            word_count = len(text.split())
                            logger.info(f"Successfully extracted {word_count} words from {len(pdf_reader.pages)} pages using PyPDF2")
                            result["text"] = text.strip()
                            return result
                        
                        logger.info("PyPDF2 text extraction completed but no text was found, attempting OCR...")
                        
                    except Exception as pdf_error:
                        error_msg = f"PyPDF2 processing failed: {str(pdf_error)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        result["processing_errors"].append(error_msg)
                        
            except Exception as e:
                error_msg = f"Error processing PDF: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                result["processing_errors"].append(error_msg)
                
            # Clean up the temporary file if it exists
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"Warning: Could not delete temporary file {temp_file.name}: {str(e)}")
            
            # Try to extract tables using tabula if available
            try:
                # Check if tabula is available
                if not hasattr(tabula, 'read_pdf_with_pandas'):
                    raise ImportError("Tabula is not properly installed or configured")
                    
                tables = tabula.read_pdf_with_pandas(
                    input_path=file_path,
                    pages='all',
                    multiple_tables=True,
                    pandas_options={'header': None}
                )
                
                for i, table in enumerate(tables, 1):
                    if not table.empty:
                        result["tables"][f"Table_{i}"] = table
                        
            except Exception as table_error:
                logger.error(f"Table extraction failed: {str(table_error)}")
                result["processing_errors"].append(f"Table extraction failed: {str(table_error)}")
                
        except Exception as e:
            error_msg = f"Unexpected error during PDF processing: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            result["processing_errors"].append(error_msg)
            
        # Log processing details
        file_name = getattr(file, 'name', 'unknown_file')
        logger.info(f"PDF processing completed for {file_name}")
        
        if result["processing_errors"]:
            logger.warning(f"PDF processing completed with {len(result['processing_errors'])} errors")
            for i, error in enumerate(result["processing_errors"], 1):
                logger.warning(f"  Error {i}: {error}")
        
        # If we have text or tables, consider it a success
        if result["text"] or result["tables"]:
            logger.info("PDF processing successful - content extracted")
            return result
            
        # If we get here, all processing attempts failed
        if not result["processing_errors"]:
            error_msg = "Failed to extract any content from PDF (no specific error reported)"
            result["processing_errors"].append(error_msg)
            logger.error(error_msg)
        
        return result

    def _extract_tables_from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract multiple tables from a single DataFrame by identifying empty rows as separators.
        Returns a dictionary of tables with their data.
        """
        tables = {}
        
        # Find rows where all values are NaN or empty (potential table separators)
        mask = df.isna().all(axis=1) | (df == '').all(axis=1)
        separator_indices = df.index[mask].tolist()
        
        # Add start and end indices
        start_indices = [0] + [i + 1 for i in separator_indices if i + 1 < len(df)]
        end_indices = separator_indices + [len(df)]
        
        # If no separators found, treat entire DataFrame as one table
        if not start_indices:
            return {"Table_1": df}
            
        # Extract tables between separators
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start >= end:
                continue
                
            table_df = df.iloc[start:end].reset_index(drop=True)
            if not table_df.empty and not table_df.isna().all().all():
                table_name = f"Table_{i+1}"
                tables[table_name] = table_df
                
        return tables

    def _process_excel(self, file) -> Dict[str, Any]:
        """Extract text and tables from Excel files with support for multiple sheets and tables."""
        result = {"text": "", "sheets": {}, "tables": {}, "raw_data": {}}
        
        try:
            # Ensure we have a file-like object with seek support
            if hasattr(file, 'seek'):
                file.seek(0)
                
            excel_file = pd.ExcelFile(file)
            
            # Process each sheet
            for sheet_name in excel_file.sheet_names:
                try:
                    # First try reading with headers
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
                    
                    # If no data, try without headers
                    if df.empty or (len(df.columns) == 1 and df.iloc[:, 0].isna().all()):
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, engine='openpyxl')
                    
                    # Clean up the DataFrame
                    df = df.dropna(how='all').reset_index(drop=True)
                    
                    # If still no data, skip this sheet
                    if df.empty:
                        continue
                        
                    # Clean column names and handle unnamed columns
                    df.columns = [f'Column_{i}' if (pd.isna(col) or str(col).startswith('Unnamed:')) 
                                else str(col).strip() for i, col in enumerate(df.columns, 1)]
                    
                    # Clean up data - replace NaN with empty strings for display
                    df = df.fillna('')
                    
                    # Store raw data for this sheet
                    result["raw_data"][sheet_name] = df.to_dict('records')
                    
                    # Store sheet metadata
                    result["sheets"][sheet_name] = {
                        'shape': df.shape,
                        'columns': [str(col) for col in df.columns],
                        'sample_data': df.head(5).to_dict('records')
                    }
                    
                    # Process the sheet as a table
                    table_name = f"{sheet_name}"
                    table_data = {
                        'data': df.to_dict('records'),
                        'columns': [str(col) for col in df.columns],
                        'shape': df.shape,
                        'sheet': sheet_name,
                        'table_name': table_name
                    }
                    
                    # Store the table
                    result["tables"][table_name] = table_data
                    
                    # Create a text representation of the table for embedding
                    text_representation = self._create_table_text_representation(table_data, max_rows=10)
                    
                    # Add to the main text content
                    if text_representation:
                        if result["text"]:
                            result["text"] += "\n\n" + "-"*50 + "\n\n"
                        result["text"] += f"Sheet: {sheet_name}\n" + text_representation
                    
                except Exception as sheet_error:
                    logger.error(f"Error processing sheet '{sheet_name}': {str(sheet_error)}")
                    logger.error(traceback.format_exc())
                    continue
            
            # If no content was extracted, add a message
            if not result["text"] and not result["tables"]:
                result["text"] = "Excel file contains no processable data"
            
        except Exception as e:
            error_msg = f"Error processing Excel file: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            result["error"] = error_msg
            
        return result
        
    def _create_table_text_representation(self, table_data: Dict[str, Any], max_rows: int = 10) -> str:
        """
        Create a detailed text representation of a table for embedding and display.
        
        Args:
            table_data: Dictionary containing table data and metadata
            max_rows: Maximum number of rows to include in the representation
            
        Returns:
            Formatted string representation of the table
        """
        try:
            if not table_data or 'data' not in table_data or not table_data['data']:
                return "No table data available"
                
            df = pd.DataFrame(table_data['data'])
            if df.empty:
                return "Empty table"
            
            # Clean up the data
            df = df.fillna('')
            
            # Get column names
            columns = table_data.get('columns', [])
            if not columns and hasattr(df, 'columns'):
                columns = [str(col) for col in df.columns]
            
            lines = []
            
            # Add table summary
            lines.append(f"Table: {table_data.get('table_name', 'Unnamed')} (Sheet: {table_data.get('sheet', 'N/A')})")
            lines.append(f"Dimensions: {len(df)} rows × {len(columns)} columns")
            lines.append("")
            
            # Add column information
            lines.append("### Columns:")
            col_info = []
            for i, col in enumerate(columns, 1):
                col_data = df[col]
                non_empty = col_data[col_data != ''].count()
                unique = col_data.nunique()
                sample = ", ".join([str(x) for x in col_data.head(3).tolist() if x != ''])
                if len(col_data) > 3:
                    sample += "..."
                
                col_info.append(
                    f"- {col} (Col {i}): {non_empty}/{len(df)} non-empty, "
                    f"{unique} unique values. Sample: {sample or 'N/A'}"
                )
            lines.extend(col_info)
            lines.append("")
            
            # Add data preview
            lines.append("### Data Preview:")
            
            # Create a markdown table header
            if columns:
                lines.append(" | ".join([""] + [str(i+1) for i in range(len(columns))]))
                lines.append(" | ".join(["---"] * (len(columns) + 1)))
                
                # Add column headers
                lines[-1] = "Row | " + " | ".join(columns)
                lines.append("--- | " + " | ".join(["---"] * len(columns)))
            
            # Add data rows (limit to max_rows)
            for i, (_, row) in enumerate(df.head(max_rows).iterrows(), 1):
                row_values = [str(i)]  # Add row number
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value) or value == '':
                        row_values.append('')
                    else:
                        # Truncate long values for better readability
                        cell_content = str(value)
                        if len(cell_content) > 50:
                            cell_content = cell_content[:47] + "..."
                        row_values.append(cell_content)
                lines.append(" | ".join(row_values))
            
            # Add summary if there are more rows
            if len(df) > max_rows:
                lines.append(f"... and {len(df) - max_rows} more rows")
            
            # Add data statistics
            lines.append("")
            lines.append("### Statistics:")
            lines.append(f"- Total Rows: {len(df)}")
            lines.append(f"- Total Columns: {len(columns)}")
            
            # Add column-wise statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                lines.append("\nNumeric Columns:")
                for col in numeric_cols:
                    stats = df[col].describe()
                    lines.append(
                        f"- {col}: "
                        f"min={stats.get('min', 'N/A')}, "
                        f"max={stats.get('max', 'N/A')}, "
                        f"mean={stats.get('mean', 'N/A'):.2f}, "
                        f"std={stats.get('std', 'N/A'):.2f}"
                    )
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error creating table text representation: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def _process_csv(self, file) -> Dict[str, Any]:
        """Extract data from CSV files."""
        result = {"text": "", "tables": {}}
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(io.StringIO(file.read().decode(encoding)))
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is not None and not df.empty:
                result["tables"]["CSV_Data"] = df
                result["text"] = f"CSV file with {len(df)} rows and {len(df.columns)} columns"
                
        except Exception as e:
            print(f"Error processing CSV file: {str(e)}")
            
        return result

    def _process_text(self, file) -> Dict[str, Any]:
        """Extract text from plain text files."""
        try:
            content = file.getvalue().decode('utf-8')
            return {"text": content, "tables": {}}
        except Exception as e:
            print(f"Error reading text file: {str(e)}")
            return {"text": "", "tables": {}}