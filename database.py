from sqlalchemy import create_engine, event, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)  # Force override any existing variables

# Get database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Log the database URL (without password for security)
safe_url = DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL
logger.info(f"Connecting to database: postgresql://...@{safe_url}")

# Ensure the connection string is in the correct format
if not DATABASE_URL.startswith('postgresql'):
    if DATABASE_URL.startswith('postgres://'):
        # Convert postgres:// to postgresql://
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    else:
        raise ValueError("Invalid DATABASE_URL format. Must start with postgresql://")

# Parse the URL to extract components
from sqlalchemy.engine import make_url
url = make_url(DATABASE_URL)

# Add SSL configuration for Neon DB
if 'neon.tech' in url.host:
    # Ensure we have the required SSL parameters
    if 'sslmode' not in str(url.query):
        url = url.update_query_dict({
            'sslmode': 'require',
            'channel_binding': 'require'
        })
        DATABASE_URL = str(url)
        logger.info("Added SSL parameters to database connection URL")
    
    # Verify the host is correct (should be the neon.tech endpoint)
    if not hasattr(url, 'host') or not url.host or 'neon.tech' not in url.host:
        raise ValueError("Invalid database host. Make sure you're using the correct Neon DB endpoint")

# Create SQLAlchemy engine with connection pooling and SSL configuration
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    echo=True,  # Enable SQL query logging for debugging
    # Add SSL configuration
    connect_args={
        'sslmode': 'require',
        'sslrootcert': None,  # Let the system use default CA certificates
        'target_session_attrs': 'read-write'
    } if 'neon.tech' in DATABASE_URL else {}
)

# Create a scoped session factory
db_session = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False
    )
)

# Base class for models
Base = declarative_base()
Base.query = db_session.query_property()

def init_db():
    """
    Initialize the database by creating all tables if they don't exist.
    Performs a safe initialization without dropping existing tables.
    """
    # Import models to ensure they are registered with SQLAlchemy
    from models import Document, DocumentChunk, ChatMessage
    
    try:
        # Test the connection first
        with engine.connect() as conn:
            logger.info(f"Successfully connected to database: {engine.url}")
            
            # Check if tables exist
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            required_tables = {'documents', 'document_chunks', 'chat_messages'}
            
            # Only create tables that don't exist
            tables_to_create = required_tables - set(existing_tables)
            
            if tables_to_create:
                logger.info(f"Creating missing tables: {', '.join(tables_to_create)}")
                # Create only the missing tables
                Base.metadata.create_all(bind=engine, tables=[
                    table for table in Base.metadata.sorted_tables 
                    if table.name in tables_to_create
                ])
                logger.info("Database initialization complete")
            else:
                logger.info("All required tables already exist. No changes made.")
                
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def get_db():
    """
    Dependency to get DB session.
    Use this in FastAPI/other web frameworks to get a database session.
    
    Yields:
        SQLAlchemy session
    """
    db = db_session()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()

# Add PostgreSQL-specific settings if using PostgreSQL
if DATABASE_URL.startswith('postgresql'):
    @event.listens_for(engine, 'connect')
    def set_search_path(dbapi_connection, connection_record):
        """Set the search path for PostgreSQL connections."""
        cursor = dbapi_connection.cursor()
        cursor.execute('SET search_path TO public')
        cursor.close()

