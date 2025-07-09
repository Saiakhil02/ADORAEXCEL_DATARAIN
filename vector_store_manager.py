import os
import json
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone_ser import PineconeService

class VectorStoreManager:
    """
    Manages Pinecone vector store operations using the enhanced PineconeService.
    """
    def __init__(self, index_name: str = "default-index"):
        """
        Initialize the VectorStoreManager with PineconeService.
        
        Args:
            index_name: Name of the Pinecone index to use
        """
        self.index_name = index_name.lower()
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        # Use the enhanced PineconeService
        self.pinecone_service = PineconeService()
        
        # For backward compatibility with the existing interface
        self.vectorstore = None
        
    def _documents_to_table_data(self, documents: List[Document]) -> List[Dict]:
        """Convert LangChain documents to table data format for PineconeService."""
        table_data = []
        for i, doc in enumerate(documents):
            row = {
                "document_id": f"doc_{i}",
                "content": doc.page_content,
                "metadata": json.dumps(doc.metadata),
                "source": doc.metadata.get("source", "unknown")
            }
            table_data.append(row)
        return table_data
            
    def create_vector_store(self, 
                          documents: List[Document], 
                          namespace: str = "default",
                          organization_id: int = 1,
                          excel_file_id: int = 1) -> bool:
        """
        Create a new vector store from documents using PineconeService.
        
        Args:
            documents: List of Document objects
            namespace: For compatibility (not used in new implementation)
            organization_id: Organization ID for filtering
            excel_file_id: File ID for tracking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                return False
                
            # Convert documents to table data format
            table_data = self._documents_to_table_data(documents)
            
            # Index using PineconeService
            vector_ids = self.pinecone_service.index_table(
                organization_id=organization_id,
                excel_file_id=excel_file_id,
                sheet_name=namespace or "default",
                table_name="documents",
                table_data=table_data,
                file_name=f"file_{excel_file_id}"
            )
            
            return len(vector_ids) > 0
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def add_documents(self, 
                     documents: List[Document], 
                     namespace: str = "default",
                     organization_id: int = 1,
                     excel_file_id: int = 1,
                     **kwargs) -> bool:
        """
        Add documents to the vector store using PineconeService.
        
        Args:
            documents: List of Document objects to add
            namespace: For compatibility (used as sheet_name)
            organization_id: Organization ID for filtering
            excel_file_id: File ID for tracking
            **kwargs: Additional keyword arguments (for compatibility)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not documents:
            return False
            
        try:
            # Convert documents to table data format
            table_data = self._documents_to_table_data(documents)
            
            # Index using PineconeService
            vector_ids = self.pinecone_service.index_table(
                organization_id=organization_id,
                excel_file_id=excel_file_id,
                sheet_name=namespace or "default",
                table_name="documents",
                table_data=table_data,
                file_name=f"file_{excel_file_id}"
            )
            
            return len(vector_ids) > 0
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False
            
    def update_vector_store(self, 
                          new_documents: List[Document],
                          namespace: str = "default",
                          organization_id: int = 1,
                          excel_file_id: int = 1,
                          **kwargs) -> bool:
        """
        Update an existing vector store with new documents.
        
        Args:
            new_documents: List of new Document objects
            namespace: For compatibility (used as sheet_name)
            organization_id: Organization ID for filtering
            excel_file_id: File ID for tracking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not new_documents:
                return False
            
            # Convert documents to table data format
            table_data = self._documents_to_table_data(new_documents)
            
            # Index using PineconeService
            vector_ids = self.pinecone_service.index_table(
                organization_id=organization_id,
                excel_file_id=excel_file_id,
                sheet_name=namespace or "default",
                table_name="documents",
                table_data=table_data,
                file_name=f"file_{excel_file_id}"
            )
            
            return len(vector_ids) > 0
            
        except Exception as e:
            print(f"Error updating vector store: {str(e)}")
            return False
    
    def get_retriever(self, 
                     k: int = 4, 
                     namespace: str = "default",
                     organization_id: int = 1,
                     excel_file_id: Optional[int] = None):
        """
        Get a retriever that uses PineconeService for document retrieval.
        
        Args:
            k: Number of documents to retrieve
            namespace: For compatibility
            organization_id: Organization ID for filtering
            excel_file_id: Optional file ID to limit search
            
        Returns:
            PineconeRetriever object
        """
        try:
            return PineconeRetriever(
                pinecone_service=self.pinecone_service,
                organization_id=organization_id,
                excel_file_id=excel_file_id,
                k=k
            )
        except Exception as e:
            print(f"Error getting retriever: {str(e)}")
            return None
    
    def delete_vector_store(self, 
                           namespace: str = None,
                           organization_id: int = 1,
                           excel_file_id: Optional[int] = None) -> bool:
        """
        Delete vectors from Pinecone using PineconeService.
        
        Args:
            namespace: For compatibility (not used)
            organization_id: Organization ID for filtering
            excel_file_id: Optional file ID to delete specific file data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if excel_file_id:
                # Delete specific file data
                return self.pinecone_service.delete_file_data(organization_id, excel_file_id)
            else:
                # Delete all organization data
                return self.pinecone_service.delete_organization_data(organization_id)
                
        except Exception as e:
            print(f"Error deleting vector store: {str(e)}")
            return False

    def search_documents(self, 
                        query: str,
                        organization_id: int = 1,
                        excel_file_id: Optional[int] = None,
                        k: int = 4) -> List[Document]:
        """
        Search for documents using PineconeService.
        
        Args:
            query: Search query
            organization_id: Organization ID for filtering
            excel_file_id: Optional file ID to limit search
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        try:
            # Search using PineconeService
            results = self.pinecone_service.search_tables(
                organization_id=organization_id,
                query=query,
                top_k=k,
                excel_file_id=excel_file_id
            )
            
            # Convert results back to LangChain Documents
            documents = []
            for result in results:
                # Extract content and metadata from table_data
                content = result.get('table_data', [])
                if content:
                    # Convert table data to readable content
                    content_str = "\n".join([
                        f"Row {i+1}: " + " | ".join([f"{k}: {v}" for k, v in row.items()])
                        for i, row in enumerate(content)
                    ])
                else:
                    content_str = "No content available"
                
                metadata = {
                    "source": result.get('file_name', 'unknown'),
                    "sheet_name": result.get('sheet_name'),
                    "table_name": result.get('table_name'),
                    "score": result.get('score'),
                    "organization_id": result.get('organization_id'),
                    "excel_file_id": result.get('excel_file_id'),
                    "table_id": result.get('table_id')
                }
                
                doc = Document(page_content=content_str, metadata=metadata)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []


# Custom retriever class for PineconeService integration
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

class PineconeRetriever(BaseRetriever):
    """Custom retriever that uses PineconeService for document retrieval."""
    
    def __init__(self, 
                 pinecone_service: PineconeService,
                 organization_id: int,
                 excel_file_id: Optional[int] = None,
                 k: int = 4):
        super().__init__()
        self.pinecone_service = pinecone_service
        self.organization_id = organization_id
        self.excel_file_id = excel_file_id
        self.k = k
    
    def _get_relevant_documents(self, 
                              query: str, 
                              *, 
                              run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Retrieve relevant documents using PineconeService."""
        try:
            # Search using PineconeService
            results = self.pinecone_service.search_tables(
                organization_id=self.organization_id,
                query=query,
                top_k=self.k,
                excel_file_id=self.excel_file_id
            )
            
            # Convert results to LangChain Documents
            documents = []
            for result in results:
                # Extract content from table_data
                content = result.get('table_data', [])
                if content:
                    # Convert table data to readable content
                    content_str = "\n".join([
                        f"Row {i+1}: " + " | ".join([f"{k}: {v}" for k, v in row.items()])
                        for i, row in enumerate(content)
                    ])
                else:
                    content_str = "No content available"
                
                metadata = {
                    "source": result.get('file_name', 'unknown'),
                    "sheet_name": result.get('sheet_name'),
                    "table_name": result.get('table_name'),
                    "score": result.get('score'),
                    "organization_id": result.get('organization_id'),
                    "excel_file_id": result.get('excel_file_id'),
                    "table_id": result.get('table_id')
                }
                
                doc = Document(page_content=content_str, metadata=metadata)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
