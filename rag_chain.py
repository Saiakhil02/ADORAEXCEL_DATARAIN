import os
from typing import List, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from vector_store_manager import VectorStoreManager

class RAGChain:
    """
    Handles the RAG (Retrieval-Augmented Generation) functionality
    including document indexing and question answering using VectorStoreManager.
    """
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self.current_store = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model."""
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def _initialize_qa_chain(self):
        """Initialize the QA chain with the current retriever."""
        if not self.llm:
            self._initialize_llm()
            
        if not self.retriever:
            # Create a dummy retriever that returns an empty list
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
            from typing import List
            
            class DummyRetriever(BaseRetriever):
                def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                    return []
                    
            self.retriever = DummyRetriever()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def update_knowledge_base(self, 
                             content: Union[str, Dict], 
                             namespace: str,
                             organization_id: int = 1,
                             excel_file_id: int = 1):
        """
        Update the knowledge base with new content using PineconeService.
        
        Args:
            content: Either a string of text or a dictionary with 'text' and 'tables' keys
            namespace: Namespace for the documents (usually file name)
            organization_id: Organization ID for filtering (default: 1)
            excel_file_id: File ID for tracking (default: 1)
        """
        try:
            # Clean the namespace to ensure it's a valid store name
            store_name = os.path.splitext(str(namespace).strip())[0]
            if not store_name:
                raise ValueError("Invalid namespace provided")
                
            # Clear any existing vector store from memory
            self.clear_current_store()
            
            # Prepare documents for indexing
            documents = []
            if isinstance(content, dict):
                # Handle both text and tables
                if 'text' in content and content['text']:
                    documents.append(Document(page_content=content['text']))
                if 'tables' in content and content['tables']:
                    for table_name, table_data in content['tables'].items():
                        if 'data' in table_data and table_data['data']:
                            table_text = f"Table {table_name}: {table_data.get('description', '')}"
                            documents.append(Document(page_content=table_text, 
                                                   metadata={'source': namespace, 'table': table_name}))
            else:
                # Single text content
                documents = [Document(page_content=str(content))]
            
            # Add documents using the new interface
            if documents:
                success = self.vector_store.add_documents(
                    documents, 
                    namespace=store_name,
                    organization_id=organization_id,
                    excel_file_id=excel_file_id
                )
                
                if not success:
                    raise ValueError(f"Failed to create vector store for {namespace}")
            
            # Update the current store reference
            self.current_store = store_name
            
            # Get the retriever from the vector store with organization filtering
            self.retriever = self.vector_store.get_retriever(
                k=50,
                namespace=store_name,
                organization_id=organization_id,
                excel_file_id=excel_file_id
            )
            
            # Reinitialize the QA chain with the new retriever
            self._initialize_qa_chain()
            
            if not self.retriever:
                print("Warning: No retriever available for the vector store")
                
        except Exception as e:
            print(f"Error updating knowledge base: {str(e)}")
            raise
            
    def clear_current_store(self):
        """Clear the current vector store from memory."""
        self.retriever = None
        self.qa_chain = None
        if hasattr(self.vector_store, 'vectorstore'):
            self.vector_store.vectorstore = None
            
    def delete_store(self, namespace: str, organization_id: int = 1, excel_file_id: Optional[int] = None) -> bool:
        """
        Delete a vector store by namespace using PineconeService.
        
        Args:
            namespace: The namespace of the store to delete
            organization_id: Organization ID for filtering (default: 1)
            excel_file_id: Optional file ID to delete specific file data
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            store_name = os.path.splitext(str(namespace).strip())[0]
            if not store_name:
                return False
                
            # Clear from memory if it's the current store
            if self.current_store == store_name:
                self.clear_current_store()
                self.current_store = None
                
            # Delete using PineconeService
            return self.vector_store.delete_vector_store(
                namespace=store_name,
                organization_id=organization_id,
                excel_file_id=excel_file_id
            )
            
        except Exception as e:
            print(f"Error deleting store {namespace}: {str(e)}")
            return False

    def get_response(self, query: str) -> Dict[str, Any]:
        """
        Get a response for the given query using the RAG model.
        
        Args:
            query: User's question or query
            
        Returns:
            Dictionary containing the response, visualization instructions, and source documents
        """
        try:
            # Ensure QA chain is initialized
            if not self.qa_chain:
                self._initialize_qa_chain()
                
            if not self.retriever or not self.current_store:
                return {
                    "answer": "Please upload and process a document first.",
                    "sources": [],
                    "visualization_instructions": None
                }
                
            # Get relevant documents using the retriever
            docs = self.retriever.get_relevant_documents(query)
            
            # If no relevant documents found
            if not docs:
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "visualization_instructions": None
                }
            
            # Format the context from documents
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                  for i, doc in enumerate(docs)])
            
            # Check if the query is asking for a visualization
            visualization_keywords = ['chart', 'graph', 'plot', 'visualize', 'show me', 'display', 
                                   'bar', 'line', 'pie', 'scatter', 'histogram', 'visualization']
            
            is_visualization_query = any(keyword in query.lower() for keyword in visualization_keywords)
            
            # Create a prompt with the context and query
            if is_visualization_query:
                prompt = f"""You are an AI assistant that helps analyze data and create visualizations. 
                The user has asked a question that requires a visualization. Your task is to:
                
                1. Analyze the data in the context
                2. Determine the most appropriate type of visualization
                3. Provide clear instructions for creating the visualization
                4. Include a brief explanation of what the visualization shows
                
                Context:
                {context}
                
                User's question: {query}
                
                Please respond with a JSON object containing these fields:
                {{
                    "answer": "Your detailed response explaining the visualization and insights",
                    "visualization_instructions": {{
                        "type": "bar|line|scatter|pie|histogram|box|heatmap",
                        "data_columns": ["column1", "column2", ...],
                        "x_axis": "column_name",
                        "y_axis": ["column1", "column2", ...],
                        "title": "Descriptive title for the visualization",
                        "x_label": "X-axis label",
                        "y_label": "Y-axis label",
                        "description": "Brief description of what the visualization shows"
                    }}
                }}
                """
            else:
                prompt = f"""You are a helpful assistant that answers questions based on the provided context.
                
                Context:
                {context}
                
                Question: {query}
                
                Please provide a detailed answer based on the context above. If the answer isn't in the context, say you don't know.
                """
            
            # Get response from the LLM
            response = self.llm.invoke(prompt)
            
            try:
                # Try to parse the response as JSON if it's a visualization query
                if is_visualization_query:
                    import json
                    try:
                        # Clean the response to handle markdown code blocks
                        response_text = response.content.strip()
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].strip()
                            if response_text.startswith('json'):
                                response_text = response_text[4:].strip()
                        
                        response_data = json.loads(response_text)
                        return {
                            "answer": response_data.get("answer", "Here's the visualization you requested."),
                            "visualization_instructions": response_data.get("visualization_instructions"),
                            "sources": [
                                {
                                    "content": doc.page_content,
                                    "metadata": doc.metadata
                                }
                                for doc in docs
                            ]
                        }
                    except json.JSONDecodeError:
                        # If JSON parsing fails, return the response as is
                        pass
                
                # Default response format for non-visualization queries or if JSON parsing fails
                return {
                    "answer": response.content,
                    "visualization_instructions": None,
                    "sources": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in docs
                    ]
                }
                
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                return {
                    "answer": response.content,
                    "visualization_instructions": None,
                    "sources": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in docs
                    ]
                }
            
        except Exception as e:
            print(f"Error getting response: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error processing your request: {str(e)}",
                "sources": []
            }

    def clear_knowledge_base(self):
        """Clear the current knowledge base."""
        self.initialize_qa_chain()
