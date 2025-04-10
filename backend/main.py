from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from paper_processor import PaperProcessor
import shutil
import logging

app = FastAPI(title="Paper Analysis System")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directory for uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create PaperProcessor instance
paper_processor = PaperProcessor()

logger = logging.getLogger(__name__)

class Question(BaseModel):
    query: str
    paper_id: Optional[str] = None

class PaperSummary(BaseModel):
    summary: str
    file_path: str
    total_pages: int

@app.post("/upload", response_model=PaperSummary)
async def upload_paper(file: UploadFile = File(...)):
    """Upload and analyze paper file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process paper
    try:
        result = paper_processor.process_paper(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: Question):
    """Process questions about the paper"""
    try:
        answer = paper_processor.ask_question(question.query, question.paper_id)
        if not answer:
            raise HTTPException(status_code=500, detail="Failed to generate an answer.")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error in ask_question endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers")
async def list_papers():
    """Get list of stored papers"""
    try:
        papers = []
        for filename in os.listdir(UPLOAD_DIR):
            if filename.endswith('.pdf'):
                papers.append({
                    "id": filename,
                    "name": filename,
                    "path": os.path.join(UPLOAD_DIR, filename)
                })
        return {"papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Delete a paper by ID"""
    try:
        file_path = os.path.join(UPLOAD_DIR, paper_id)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Paper not found.")
        
        # Delete the file
        os.remove(file_path)
        
        # Also delete from vector store if it exists
        try:
            paper_processor.vector_store.delete(ids=[paper_id])
        except Exception as e:
            logger.warning(f"Error deleting from vector store: {str(e)}")
        
        return {"message": "Paper deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting paper: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting paper: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 