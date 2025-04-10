# Thesis Assistant

A web-based application that helps users analyze academic papers using AI. The application uses Ollama's DeepSeek model for question answering and document analysis.

## Features

- PDF Document Processing
  - Upload and process academic papers in PDF format
  - Document vectorization for semantic search
  
- Analysis
  - Generate summaries of uploaded papers
  - Ask questions about paper content

## Technical Stack

### Backend
- Python with FastAPI
- LangChain for AI orchestration
- Sentence Transformers for document embeddings
- ChromaDB for vector storage
- PDF processing with PyPDF and OCR capabilities

### Frontend
- React with TypeScript

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- Ollama installed with DeepSeek-r1:8b model

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   
## Notes

- The system uses Ollama's DeepSeek-r1:8b model for text generation and can change the model
- Should download Ollma's local model
- Maximum context length is limited to 1500 characters for optimal performance

## Architecture

The application follows a client-server architecture:

- **Frontend**: React-based UI that communicates with the backend API
- **Backend**: FastAPI server that handles:
  - PDF processing and text extraction
  - Vector embeddings generation using Sentence Transformers
  - Semantic search using ChromaDB
  - Question answering using Ollama's DeepSeek model
