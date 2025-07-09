import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone_ser import PineconeService

class VectorStoreManager:
    """
    Manages the Pinecone vector store for document embeddings and similarity search.
    Uses the enhanced PineconeService for better organization-based filtering.
    """
    
    def __init__(self, index_name: str = "default-index"):
        """
        Initialize the VectorStoreManager.
        
        Args:
            index_name: Name of the index (maintained for compatibility)
        """
        self.index_name = index_name.lower()
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        # Use the enhanced PineconeService
        self.pinecone_service = PineconeService()
        self.vectorstore = None  # For backward compatibility
        
    def _documents_to_table_data(self, documents: List[Document]) -> List[Dict]:
        """Convert LangChain documents to table data format for PineconeService."""
        import json
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
            
    def add_documents(self, 
                     documents: List[Document], 
                     namespace: str = "default",
                     organization_id: int = 1,
                     excel_file_id: int = 1) -> bool:
        """
        Add documents to the vector store using PineconeService.
        
        Args:
            documents: List of Document objects to add
            namespace: For compatibility (used as sheet_name)
            organization_id: Organization ID for filtering
            excel_file_id: File ID for tracking
            
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
    
    def similarity_search(
        self, 
        query: str, 
        namespace: str = "default",
        organization_id: int = 1,
        excel_file_id: Optional[int] = None,
        k: int = 4, 
        **kwargs
    ) -> List[Document]:
        """
        Perform a similarity search using PineconeService.
        
        Args:
            query: The query string
            namespace: For compatibility
            organization_id: Organization ID for filtering
            excel_file_id: Optional file ID to limit search
            k: Number of results to return
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            List of Document objects most similar to the query
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
            print(f"Error in similarity search: {str(e)}")
            return []

    def clear(self, 
             organization_id: int = 1,
             excel_file_id: Optional[int] = None) -> None:
        """
        Clear vectors from the vector store using PineconeService.
        
        Args:
            organization_id: Organization ID for filtering
            excel_file_id: Optional file ID to clear specific file data
        """
        try:
            if excel_file_id:
                self.pinecone_service.delete_file_data(organization_id, excel_file_id)
            else:
                self.pinecone_service.delete_organization_data(organization_id)
            self.vectorstore = None
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")

# Global instance
vector_store_manager = VectorStoreManager()
