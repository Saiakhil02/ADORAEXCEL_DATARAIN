from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from database import Base

class Document(Base):
    """Model for storing document metadata."""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(Text, nullable=False)  # Using Text instead of String for unlimited length
    file_type = Column(Text)  # Using Text instead of String for unlimited length
    file_size = Column(Integer)  # Size in bytes
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processing_errors = Column(Text, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document")
    chat_messages = relationship("ChatMessage", back_populates="document")

class DocumentChunk(Base):
    """Model for storing document chunks with embeddings."""
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(JSON, nullable=True)  # Store vector embeddings as JSON
    chunk_metadata = Column('metadata', JSON, nullable=True)   # Additional metadata as JSON
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class ChatMessage(Base):
    """Model for storing chat message history."""
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), nullable=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_metadata = Column('metadata', JSON, nullable=True)  # For storing additional data like visualization
    
    # Relationships
    document = relationship("Document", back_populates="chat_messages")

# Add any additional models here as needed
