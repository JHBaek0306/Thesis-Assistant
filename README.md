# Thesis Assistant

A web-based application that helps users analyze academic papers using AI. The application uses Ollama's DeepSeek model for question answering and document analysis.

## Features

- PDF Document Processing
  - Upload and process academic papers in PDF format
  - Automatic text extraction with OCR support for image-based PDFs
  - Document vectorization for semantic search
  
- AI-Powered Analysis
  - Generate summaries of uploaded papers
  - Ask questions about paper content
  - Semantic search across uploaded documents
  - Uses DeepSeek-r1:8b model through Ollama for advanced language understanding
  
- User Interface
  - Modern React-based frontend
  - Drag-and-drop file upload
  - Real-time question answering
  - Paper management with delete functionality

## Technical Stack

### Backend
- Python with FastAPI
- LangChain for AI orchestration
- Sentence Transformers for document embeddings
- ChromaDB for vector storage
- PDF processing with PyPDF and OCR capabilities

### Frontend
- React with TypeScript
- Material-UI components
- Axios for API communication

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

## Usage

1. Access the application at `http://localhost:3000`
2. Upload PDF documents using the drag-and-drop interface
3. View uploaded papers in the papers list
4. Ask questions about the papers using the question input field
5. Delete papers using the delete button when needed

## Notes

- The system uses Ollama's DeepSeek-r1:8b model for text generation
- OCR functionality is available for image-based PDFs
- All AI thinking processes are logged in `backend/logs/thinking_process.log`
- Maximum context length is limited to 1500 characters for optimal performance

## Architecture

The application follows a client-server architecture:

- **Frontend**: React-based UI that communicates with the backend API
- **Backend**: FastAPI server that handles:
  - PDF processing and text extraction
  - Vector embeddings generation using Sentence Transformers
  - Semantic search using ChromaDB
  - Question answering using Ollama's DeepSeek model

## Development

### Adding New Features
1. Implement backend functionality in the appropriate Python modules
2. Create or update API endpoints in `main.py`
3. Implement frontend components and connect them to the API

### Testing
- Backend tests can be run with:
  ```
  cd backend
  pytest
  ```
- Frontend tests can be run with:
  ```
  cd frontend
  npm test
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 