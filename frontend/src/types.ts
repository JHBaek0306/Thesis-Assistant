export interface Paper {
  id: string;
  name: string;
  path: string;
}

export interface PaperSummary {
  summary: string;
  file_path: string;
  total_pages: number;
}

export interface Question {
  query: string;
  paper_id?: string;
}

export interface QuestionResponse {
  answer: string;
}

export interface PapersResponse {
  papers: Paper[];
} 