"""
Pinecone Vector Database Service for Organization-based Excel Table Search
"""
import os
import json
import hashlib
import uuid
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PineconeService:
    """Service class for handling Pinecone vector operations with organization filtering."""
    
    def __init__(self):
        """Initialize Pinecone and OpenAI clients."""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "finance-tables")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        
        # print(f"🔍 Pinecone Configuration:")
        # print(f"  - API Key: {'✅ Set' if self.pinecone_api_key else '❌ Missing'}")
        # print(f"  - OpenAI Key: {'✅ Set' if self.openai_api_key else '❌ Missing'}")
        # print(f"  - Index Name: {self.index_name}")
        # print(f"  - Environment: {self.environment}")
        # print(f"  - Embedding Model: {self.embedding_model}")
        # print(f"  - Embedding Dimension: {self.embedding_dimension}")
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            # print("✅ Pinecone client initialized")
        except Exception as e:
            # print(f"❌ Failed to initialize Pinecone client: {e}")
            raise
        
        # Initialize OpenAI
        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            # print("✅ OpenAI client initialized")
        except Exception as e:
            # print(f"❌ Failed to initialize OpenAI client: {e}")
            raise
        
        # Initialize or get index
        self.index = self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            # print(f"🔍 Existing indexes: {existing_indexes}")
            
            if self.index_name not in existing_indexes:
                # print(f"📝 Creating new index: {self.index_name}")
                # Create index with configurable dimensions for OpenAI embeddings
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                # print(f"✅ Created new Pinecone index: {self.index_name}")
                
                # Wait for index to be ready
                import time
                # print("⏳ Waiting for index to be ready...")
                time.sleep(20)  # Increased wait time
            else:
                # print(f"✅ Index {self.index_name} already exists")
                pass
            
            # Connect to index
            # print(f"🔌 Connecting to index: {self.index_name}")
            index = self.pc.Index(self.index_name)
            
            # Test connection by getting stats
            try:
                stats = index.describe_index_stats()
                # print(f"✅ Connected to Pinecone index: {self.index_name}")
                # print(f"  - Total vectors: {stats.total_vector_count}")
                # print(f"  - Dimension: {stats.dimension}")
                # print(f"  - Index fullness: {stats.index_fullness}")
                # print(f"  - Namespaces: {stats.namespaces}")
            except Exception as e:
                # print(f"⚠️ Connected to index but could not get stats: {e}")
                pass
                
            return index
            
        except Exception as e:
            # print(f"❌ Error initializing Pinecone index: {str(e)}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            # print(f"❌ Error generating embedding: {str(e)}")
            raise
    
    def _chunk_table_data(self, table_data: List[Dict], max_rows_per_chunk: int = 50) -> List[Dict]:
        """Split large table data into smaller chunks for better vector search."""
        chunks = []
        
        if len(table_data) <= max_rows_per_chunk:
            # Small table, return as single chunk
            return [{"chunk_index": 0, "data": table_data}]
        
        # Split into chunks
        for i in range(0, len(table_data), max_rows_per_chunk):
            chunk_data = table_data[i:i + max_rows_per_chunk]
            chunks.append({
                "chunk_index": i // max_rows_per_chunk,
                "data": chunk_data
            })
        
        return chunks
    
    def _create_searchable_text(self, table_data: List[Dict], sheet_name: str, table_name: str) -> str:
        """Create searchable text representation of table data."""
        # Create a text representation that includes:
        # 1. Sheet and table names
        # 2. Column names
        # 3. Sample of data values
        # 4. Data patterns and summary
        
        text_parts = [
            f"Sheet: {sheet_name}",
            f"Table: {table_name}"
        ]
        
        if table_data:
            # Add column information
            columns = list(table_data[0].keys())
            text_parts.append(f"Columns: {', '.join(columns)}")
            
            # Add sample data (first few rows)
            sample_size = min(5, len(table_data))
            for i, row in enumerate(table_data[:sample_size]):
                row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                text_parts.append(f"Row {i+1}: {row_text}")
            
            # Add data summary
            df = pd.DataFrame(table_data)
            text_parts.append(f"Total rows: {len(table_data)}")
            
            # Add column summaries for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col in df.columns:
                    text_parts.append(f"{col} range: {df[col].min()} to {df[col].max()}")
        
        return " | ".join(text_parts)
    
    def index_table(self, 
                   organization_id: int,
                   excel_file_id: int,
                   sheet_name: str,
                   table_name: str,
                   table_data: List[Dict],
                   file_name: str = "",
                   table_id: Optional[int] = None) -> List[str]:
        """
        Index table data in Pinecone with organization filtering.
        
        Returns:
            List of Pinecone vector IDs created
        """
        try:
            # print(f"🔍 Starting indexing for table {table_name} in organization {organization_id}")
            # print(f"  - Sheet: {sheet_name}")
            # print(f"  - File ID: {excel_file_id}")
            # print(f"  - Table ID: {table_id}")
            # print(f"  - Data rows: {len(table_data)}")
            
            if not table_data:
                # print("⚠️ No data to index")
                return []
            
            # Split table into chunks if necessary
            chunks = self._chunk_table_data(table_data)
            # print(f"📊 Split into {len(chunks)} chunks")
            vector_ids = []
            
            for chunk in chunks:
                # Create unique ID for this chunk
                chunk_id = f"org_{organization_id}_file_{excel_file_id}_sheet_{sheet_name}_table_{table_name}_chunk_{chunk['chunk_index']}"
                # Hash the ID to ensure it's valid for Pinecone
                vector_id = hashlib.md5(chunk_id.encode()).hexdigest()
                # print(f"📝 Processing chunk {chunk['chunk_index']}: {vector_id}")
                
                # Create searchable text
                searchable_text = self._create_searchable_text(
                    chunk['data'], sheet_name, table_name
                )
                # print(f"  - Searchable text length: {len(searchable_text)}")
                
                # Generate embedding
                try:
                    embedding = self._generate_embedding(searchable_text)
                    # print(f"  - Generated embedding of dimension {len(embedding)}")
                except Exception as e:
                    # print(f"❌ Failed to generate embedding: {e}")
                    raise
                
                # Prepare metadata
                metadata = {
                    "organization_id": organization_id,
                    "excel_file_id": excel_file_id,
                    "table_id": table_id if table_id is not None else 0,  # Use 0 instead of None
                    "sheet_name": sheet_name,
                    "table_name": table_name,
                    "file_name": file_name,
                    "chunk_index": chunk['chunk_index'],
                    "total_chunks": len(chunks),
                    "row_count": len(chunk['data']),
                    "columns": list(chunk['data'][0].keys()) if chunk['data'] else [],
                    "table_data": json.dumps(chunk['data'])  # Store actual data in metadata
                }
                
                # print(f"  - Metadata keys: {list(metadata.keys())}")
                
                # Upsert to Pinecone
                try:
                    # print(f"  - Attempting upsert to index: {self.index_name}")
                    # print(f"  - Vector ID: {vector_id}")
                    # print(f"  - Embedding dimension: {len(embedding)}")
                    # print(f"  - Metadata size: {len(str(metadata))} chars")
                    
                    # Use explicit namespace "org_data" to make it visible in dashboard
                    # namespace = f"org_{organization_id}"
                    # print(f"  - Using namespace: {namespace}")
                    
                    result = self.index.upsert(
                        vectors=[{
                            "id": vector_id,
                            "values": embedding,
                            "metadata": metadata
                        }]  # Explicitly set namespace
                    )
                    # print(f"  - ✅ Upsert successful: {result}")
                    
                    # Immediate verification - try to fetch the vector
                    try:
                        fetch_result = self.index.fetch([vector_id])
                        if vector_id in fetch_result.vectors:
                            # print(f"  - ✅ Vector verified in Pinecone: {vector_id}")
                            pass
                        else:
                            # print(f"  - ⚠️ Vector not found immediately after upsert: {vector_id}")
                            pass
                    except Exception as fetch_e:
                        # print(f"  - ⚠️ Could not verify vector fetch: {fetch_e}")
                        pass
                    
                    vector_ids.append(vector_id)
                except Exception as e:
                    # print(f"❌ Failed to upsert vector {vector_id}: {e}")
                    # print(f"   Error type: {type(e).__name__}")
                    # print(f"   Error details: {str(e)}")
                    raise
            
            # print(f"✅ Indexed {len(chunks)} chunks for table {table_name} in organization {organization_id}")
            # print(f"📋 Vector IDs created: {vector_ids}")
            
            # Verify by getting updated stats
            try:
                stats = self.get_index_stats()
                # print(f"📊 Updated index stats: {stats.get('total_vectors', 0)} total vectors")
            except Exception as e:
                # print(f"⚠️ Could not get updated stats: {e}")
                pass
                
            return vector_ids
            
        except Exception as e:
            # print(f"❌ Error indexing table: {str(e)}")
            raise
    
    def search_tables(self, 
                     organization_id: int,
                     query: str,
                     top_k: int = 5,
                     excel_file_id: Optional[int] = None,
                     table_id: Optional[int] = None) -> List[Dict]:
        """
        Search for relevant table chunks within an organization.
        
        Args:
            organization_id: Organization ID to filter by
            query: User's search query
            top_k: Number of results to return
            excel_file_id: Optional file ID to limit search to specific file
            table_id: Optional table ID to limit search to specific table
            
        Returns:
            List of relevant table chunks with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self._generate_embedding(query)
            
            # Prepare filter
            filter_dict = {"organization_id": organization_id}
            if excel_file_id:
                filter_dict["excel_file_id"] = excel_file_id
            if table_id:
                filter_dict["table_id"] = table_id
            
            # Search Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Process results
            relevant_chunks = []
            for match in search_results.matches:
                metadata = match.metadata
                
                # Parse table data from metadata
                try:
                    table_data = json.loads(metadata.get('table_data', '[]'))
                except json.JSONDecodeError:
                    table_data = []
                
                chunk_info = {
                    "vector_id": match.id,
                    "score": match.score,
                    "organization_id": metadata.get('organization_id'),
                    "excel_file_id": metadata.get('excel_file_id'),
                    "table_id": metadata.get('table_id'),  # NEW: Include table_id in results
                    "sheet_name": metadata.get('sheet_name'),
                    "table_name": metadata.get('table_name'),
                    "file_name": metadata.get('file_name'),
                    "chunk_index": metadata.get('chunk_index'),
                    "total_chunks": metadata.get('total_chunks'),
                    "row_count": metadata.get('row_count'),
                    "columns": metadata.get('columns', []),
                    "table_data": table_data
                }
                relevant_chunks.append(chunk_info)
            
            # print(f"✅ Found {len(relevant_chunks)} relevant chunks for organization {organization_id}")
            return relevant_chunks
            
        except Exception as e:
            # print(f"❌ Error searching tables: {str(e)}")
            raise
    
    def delete_organization_data(self, organization_id: int) -> bool:
        """Delete all vectors for an organization."""
        try:
            # Delete by filter
            self.index.delete(filter={"organization_id": organization_id})
            # print(f"✅ Deleted all data for organization {organization_id}")
            return True
        except Exception as e:
            # print(f"❌ Error deleting organization data: {str(e)}")
            return False
    
    def delete_file_data(self, organization_id: int, excel_file_id: int) -> bool:
        """Delete all vectors for a specific file in an organization."""
        try:
            # Delete by filter
            self.index.delete(filter={
                "organization_id": organization_id,
                "excel_file_id": excel_file_id
            })
            # print(f"✅ Deleted data for file {excel_file_id} in organization {organization_id}")
            return True
        except Exception as e:
            # print(f"❌ Error deleting file data: {str(e)}")
            return False
    
    def delete_table_data(self, organization_id: int, table_id: int) -> bool:
        """Delete all vectors for a specific table in an organization."""
        try:
            # Delete by filter
            self.index.delete(filter={
                "organization_id": organization_id,
                "table_id": table_id
            })
            # print(f"✅ Deleted data for table {table_id} in organization {organization_id}")
            return True
        except Exception as e:
            # print(f"❌ Error deleting table data: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the Pinecone index."""
        try:
            stats = self.index.describe_index_stats()
            # print(f"📊 Current Pinecone Index Stats:")
            # print(f"  - Index Name: {self.index_name}")
            # print(f"  - Total Vectors: {stats.total_vector_count}")
            # print(f"  - Dimension: {stats.dimension}")
            # print(f"  - Index Fullness: {stats.index_fullness}")
            # print(f"  - Namespaces: {stats.namespaces}")
            
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces
            }
        except Exception as e:
            # print(f"❌ Error getting index stats: {str(e)}")
            return {}
    
    def debug_pinecone_connection(self) -> Dict:
        """Debug Pinecone connection and configuration."""
        debug_info = {
            "api_key_set": bool(self.pinecone_api_key),
            "index_name": self.index_name,
            "environment": self.environment,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension
        }
        
        # print("\n🔍 Starting Pinecone debug...")
        # print(f"API Key: {'✅ Set' if self.pinecone_api_key else '❌ Missing'}")
        # print(f"Index Name: {self.index_name}")
        # print(f"Environment: {self.environment}")
        # print(f"Embedding Model: {self.embedding_model}")
        # print(f"Embedding Dimension: {self.embedding_dimension}")
        
        try:
            # List all indexes
            # print("\n📋 Listing available Pinecone indexes...")
            indexes = [index.name for index in self.pc.list_indexes()]
            debug_info["available_indexes"] = indexes
            debug_info["target_index_exists"] = self.index_name in indexes
            
            # print(f"Available Indexes: {indexes}")
            # print(f"Target Index Exists: {self.index_name in indexes}")
            
            if self.index_name in indexes:
                # Get index details
                # print(f"\n📊 Getting stats for index: {self.index_name}")
                stats = self.index.describe_index_stats()
                debug_info["index_stats"] = {
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "index_fullness": stats.index_fullness,
                    "namespaces": stats.namespaces
                }
                
                # print(f"Total Vectors: {stats.total_vector_count}")
                # print(f"Dimension: {stats.dimension}")
                # print(f"Index Fullness: {stats.index_fullness}")
                # print(f"Namespaces: {stats.namespaces}")
                
                # Try a simple query to test connection
                # print("\n🔎 Testing simple query...")
                
                # Get all namespaces from stats
                namespaces = stats.namespaces or {"": {"vector_count": 0}}
                debug_info["namespaces"] = namespaces
                
                # print(f"\n📊 Checking namespaces:")
                for ns_name, ns_info in namespaces.items():
                    ns_display = ns_name if ns_name else "(default)"
                    # print(f"Namespace: {ns_display}, Vectors: {ns_info.get('vector_count', 0)}")
                    
                    # Try querying each namespace
                    try:
                        # print(f"\n🔍 Testing query in namespace: {ns_display}")
                        ns_result = self.index.query(
                            vector=[0.0] * self.embedding_dimension,
                            top_k=5,
                            include_metadata=True,
                            namespace=ns_name
                        )
                        # print(f"  Found {len(ns_result.matches)} vectors in namespace {ns_display}")
                        
                        # Add to debug info
                        if not "namespace_queries" in debug_info:
                            debug_info["namespace_queries"] = {}
                        debug_info["namespace_queries"][ns_name] = len(ns_result.matches)
                    except Exception as ns_e:
                        # print(f"  ❌ Error querying namespace {ns_display}: {ns_e}")
                        pass
                
                # Default query without namespace
                # print("\n🔎 Testing query without namespace...")
                test_result = self.index.query(
                    vector=[0.0] * self.embedding_dimension,
                    top_k=5,
                    include_metadata=True
                )
                debug_info["query_test"] = "success"
                debug_info["sample_vectors"] = len(test_result.matches)
                
                # print(f"Query Test: Success")
                # print(f"Vectors Found: {len(test_result.matches)}")
                
                # If we have vectors, show details of the first one
                if test_result.matches:
                    # print("\n📝 Sample Vector Details:")
                    match = test_result.matches[0]
                    # print(f"ID: {match.id}")
                    # print(f"Score: {match.score}")
                    # print(f"Metadata Keys: {list(match.metadata.keys()) if match.metadata else 'None'}")
                    
                    # Try to fetch this vector directly
                    # print(f"\n🔍 Fetching vector {match.id} directly...")
                    try:
                        fetch_result = self.index.fetch([match.id])
                        if match.id in fetch_result.vectors:
                            # print(f"✅ Vector fetch successful")
                            pass
                        else:
                            # print(f"❌ Vector not found in fetch result")
                            pass
                    except Exception as fetch_e:
                        # print(f"❌ Vector fetch error: {fetch_e}")
                        pass
                
            else:
                # print(f"❌ Index {self.index_name} not found")
                debug_info["error"] = f"Index {self.index_name} not found"
                
        except Exception as e:
            # print(f"❌ Debug error: {e}")
            # print(f"Error type: {type(e).__name__}")
            debug_info["error"] = str(e)
            debug_info["error_type"] = type(e).__name__
        
        return debug_info 