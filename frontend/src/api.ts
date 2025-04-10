import axios from 'axios';
import { PaperSummary, Question, QuestionResponse, PapersResponse } from './types';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const uploadPaper = async (file: File): Promise<PaperSummary> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post<PaperSummary>('/upload', formData);
  return response.data;
};

export const askQuestion = async (question: Question): Promise<QuestionResponse> => {
  const response = await api.post<QuestionResponse>('/ask', question);
  return response.data;
};

export const getPapers = async (): Promise<PapersResponse> => {
  const response = await api.get<PapersResponse>('/papers');
  return response.data;
};

export const deletePaper = async (paperId: string): Promise<void> => {
  await api.delete(`/papers/${paperId}`);
}; 