# AskAdora - Document Analyzer with AI Chat

A powerful Streamlit-based application that enables users to upload and analyze various document types using natural language. The application leverages OpenAI's language models for intelligent document processing and Pinecone for efficient semantic search capabilities.

## Features

- **Multi-format Support**: Upload and analyze PDF, DOCX, XLSX, XLS, and TXT files
- **Intelligent Document Processing**: Automatic text extraction and table detection
- **Natural Language Querying**: Ask questions about your documents in plain English
- **Interactive Chat Interface**: Engage in conversations about your documents
- **Vector Database**: Pinecone-based vector store for efficient semantic search
- **Persistent Storage**: PostgreSQL database for document and chat history
- **Document Management**: View, search, and manage uploaded documents
- **Responsive Design**: Clean, modern UI that works on different screen sizes

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- PostgreSQL 12+
- Node.js and npm (for optional frontend development)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AskAdora
   ```

2. Set up a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up PostgreSQL:
   - Install PostgreSQL if not already installed
   - Create a new database named `askadora`
   - Update the `.env` file with your database credentials

5. Configure environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here

   # Database Configuration
   DATABASE_URL=postgresql://username:password@localhost:5432/askadora

   # Optional - Application settings
   DEBUG=True
   SECRET_KEY=your-secret-key-here
   MAX_FILE_SIZE_MB=200
   ```

## Project Structure

```
.
├── app.py                # Main Streamlit application
├── chatbot.py            # Chatbot implementation
├── vector_store.py       # Pinecone vector store management
├── file_processor.py     # Document processing utilities
├── database.py           # Database models and session management
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── uploads/              # Directory for uploaded files
└── static/               # Static files (CSS, JS, images)
```

## Running the Application

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Access the web interface:
   Open your browser and navigate to `http://localhost:8501`

3. Start using AskAdora:
   - Upload documents using the sidebar
   - Chat with your documents using natural language
   - Manage your uploaded documents and chat history

## How It Works

1. **Document Processing**:
   - Documents are processed based on their type (PDF, DOCX, XLSX, XLS, TXT)
   - Text is extracted and split into manageable chunks
   - Chunks are converted to vector embeddings using OpenAI's API
   - Embeddings are stored in a Pinecone vector database for efficient similarity search

2. **Query Processing**:
   - User queries are converted to vector embeddings
   - The system performs a similarity search to find relevant document chunks
   - Relevant context is passed to the OpenAI model to generate a response
   - Responses are formatted and displayed in the chat interface

## Troubleshooting

### Common Issues

- **File Upload Fails**:
  - Check file size (default limit: 200MB)
  - Verify the file type is supported
  - Ensure the uploads directory has write permissions

- **Database Connection Issues**:
  - Verify the PostgreSQL server is running
  - Check the `DATABASE_URL` in your `.env` file
  - Ensure the database user has the necessary permissions

- **OpenAI API Errors**:
  - Verify your API key is correct and has sufficient credits
  - Check the OpenAI status page for any service disruptions
  - Consider implementing rate limiting if you hit API quotas

- **Pinecone Connection Issues**:
  - Verify your Pinecone API key and environment are correct
  - Check if your Pinecone index exists and is properly configured
  - Ensure your Pinecone plan supports the required vector dimensions (1536 for OpenAI embeddings)
  - Monitor Pinecone usage quotas and limits

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
For production deployment, consider using:
- Docker for containerization
- Gunicorn or Uvicorn as a production server
- Nginx as a reverse proxy
- PostgreSQL for the database
- Redis for caching (optional)

Example Docker configuration:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [OpenAI](https://openai.com/) for the language models
- [Pinecone](https://www.pinecone.io/) for efficient similarity search and vector database
- [LangChain](https://github.com/langchain-ai/langchain) for LLM orchestration
- [SQLAlchemy](https://www.sqlalchemy.org/) for database ORM

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI and Pinecone API keys:
   ```
   OPENAI_API_KEY=your-api-key-here
   PINECONE_API_KEY=your-pinecone-api-key-here
   PINECONE_ENVIRONMENT=your-pinecone-environment-here
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Use the sidebar to upload files and interact with the chatbot

## File Support

- **PDF**: Extracts text and tables
- **Excel (xlsx, xls)**: Extracts data from all sheets
- **CSV**: Extracts tabular data
- **Text**: Extracts plain text content

## Visualization Types

The application can generate the following types of visualizations:

- Bar charts
- Line charts
- Scatter plots
- Pie charts
- Histograms
- Box plots
- Heatmaps

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit, LangChain, and OpenAI
- Uses Pinecone for efficient similarity search
- Visualization powered by Plotly
