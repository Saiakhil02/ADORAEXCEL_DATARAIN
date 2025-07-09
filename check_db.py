from database import engine, init_db, get_db
from models import Document, DocumentChunk, ChatMessage
from sqlalchemy import inspect

def check_database():
    print("=== Checking Database Connection ===")
    
    # Initialize database
    print("\nInitializing database...")
    init_db()
    
    # Check tables
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print("\nTables in database:", tables)
    
    # Check columns in each table
    for table in tables:
        print(f"\nColumns in {table}:")
        for column in inspector.get_columns(table):
            print(f"  - {column['name']} ({column['type']})")
    
    # Try to query each table
    print("\n=== Testing Queries ===")
    db = next(get_db())
    
    print("\nDocuments:")
    for doc in db.query(Document).limit(5):
        print(f"- {doc.filename} (ID: {doc.id})")
    
    print("\nChat Messages:")
    for msg in db.query(ChatMessage).limit(5):
        print(f"- {msg.role}: {msg.content[:50]}... (Doc ID: {msg.document_id})")
    
    db.close()
    print("\n=== Database Check Complete ===")

if __name__ == "__main__":
    check_database()
