from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
from dotenv import load_dotenv
import logging
import pdf2image
import pytesseract
import json
from datetime import datetime
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up file logging for Q&A interactions
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
qa_logger = logging.getLogger("qa_interactions")
qa_logger.setLevel(logging.INFO)
qa_handler = logging.FileHandler(os.path.join(log_dir, "qa_interactions.log"))
qa_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
qa_logger.addHandler(qa_handler)

load_dotenv()

class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
        
    def embed_query(self, text):
        embedding = self.model.encode(text)
        return embedding.tolist()

class PaperProcessor:
    def __init__(self):
        try:
            # Set vector_db path based on the absolute path of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vector_db_path = os.path.join(current_dir, "vector_db")
            
            logger.info("Initializing Sentence Transformer Embeddings...")
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
            logger.info("Initializing Chroma vector store...")
            self.vector_store = Chroma(
                persist_directory=vector_db_path,
                embedding_function=self.embeddings
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("Initializing Ollama DeepSeek model...")
            self.llm = Ollama(
                model="deepseek-r1:8b",
                temperature=0.1,
                num_ctx=2048,
                num_thread=2,
                repeat_penalty=1.1,
                timeout=120
            )
            logger.info("PaperProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PaperProcessor: {str(e)}")
            raise

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF. Uses OCR for image-based PDFs."""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = "\n\n".join([page.page_content.strip() for page in pages if page.page_content.strip()])
            
            if len(text.split()) > 50:  # Minimum 50 words
                logger.info("Text extracted successfully from PDF")
                return text
                
            logger.info("Attempting OCR on PDF...")
            
            images = pdf2image.convert_from_path(file_path)
            
            texts = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} with OCR")
                text = pytesseract.image_to_string(image, lang='eng')  # English only
                if text.strip():
                    texts.append(text.strip())
            
            result = "\n\n".join(texts)
            if not result.strip():
                raise ValueError("No text could be extracted from the PDF")
                
            logger.info(f"Extracted text using OCR")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def process_paper(self, file_path: str) -> Dict:
        """Process the paper and generate a summary."""
        try:
            logger.info(f"Processing paper: {file_path}")
            
            # Load PDF and extract text
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            full_text = self.extract_text_from_pdf(file_path)
            if not full_text:
                raise ValueError("No text content found in PDF file")
            
            logger.info(f"Extracted text length: {len(full_text)} characters")

            texts = self.text_splitter.split_text(full_text)
            if not texts:
                raise ValueError("Failed to split text into chunks")
            
            logger.info(f"Split into {len(texts)} text chunks")

            docs = [{"page_content": text, "metadata": {"source": file_path}} for text in texts]

            self.vector_store.add_texts(
                texts=[d["page_content"] for d in docs],
                metadatas=[d["metadata"] for d in docs]
            )
            logger.info("Added documents to vector store")

            # Generate summary prompt
            system_message = SystemMessagePromptTemplate.from_template(
                "You are an academic paper summarizer that extracts key information from research papers."
            )
            human_message = HumanMessagePromptTemplate.from_template(
                """Please provide a comprehensive summary of the following academic paper. 
                Include the following sections:
                - Main points
                - Methodology
                - Key results

                Paper text: {text}"""
            )
            summary_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            
            summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
            
            logger.info("Generating summary...")
            max_text_length = 2000
            summary_text = full_text[:max_text_length]
            response = summary_chain.invoke({"text": summary_text})
            summary = response['text']
            logger.info("Summary generated successfully")

            return {
                "summary": summary,
                "file_path": file_path,
                "total_pages": len(texts)
            }
            
        except Exception as e:
            logger.error(f"Error processing paper: {str(e)}")
            raise

    def search_papers(self, query: str, k: int = 3) -> List[Dict]:
        # Search for relevant papers in the vector store.
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            raise

    def log_qa_interaction(self, question: str, answer: str):
        # Log the Q&A interaction.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer
        }
        qa_logger.info(json.dumps(log_entry))

    def clean_answer(self, answer: str) -> str:
        # Clean the answer by removing XML-like tags and extra whitespace.
        answer = re.sub(r'<[^>]+>', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        return answer.strip()

    def ask_question(self, question: str, paper_id: Optional[str] = None) -> str:
        # Answer questions.
        try:
            logger.info(f"Received question: {question}")
            relevant_docs = self.search_papers(question, k=2)
            logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            if not relevant_docs:
                logger.warning("No relevant documents found")
                return "I'm sorry, I couldn't find any relevant content to answer your question."
            
            context = "\n".join([doc["content"] for doc in relevant_docs])
            max_context_length = 1500
            if len(context) > max_context_length:
                context = context[:max_context_length]
            
            try:
                # Create the chat prompt template
                system_message = SystemMessagePromptTemplate.from_template(
                    """You are a helpful academic assistant that provides clear and accurate answers based on the given context.
                    Always provide direct answers without any XML tags or special formatting."""
                )
                human_message = HumanMessagePromptTemplate.from_template(
                    """Based on the following context, please answer the question.
                    Provide a clear and direct answer without any XML tags or special formatting.

                    Context: {context}
                    Question: {question}"""
                )
                qa_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                
                qa_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
                
                logger.info("Generating answer...")
                response = qa_chain.invoke({"context": context, "question": question})
                answer = response['text']
                
                if not answer or not answer.strip():
                    logger.warning("LLM returned empty answer")
                    return "I'm sorry, I couldn't generate an answer. Please try a different question."
                
                answer = self.clean_answer(answer)

                self.log_qa_interaction(question, answer)
                
                logger.info("Answer generated successfully")
                
                return answer
                
            except Exception as e:
                logger.error(f"Error in QA chain: {str(e)}")
                return "I'm sorry, an error occurred while generating the answer. Please try again later."
            
        except Exception as e:
            logger.error(f"Error in ask_question: {str(e)}")
            return "I'm sorry, a system error occurred. Please contact the administrator." 