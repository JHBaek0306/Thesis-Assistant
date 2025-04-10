import React, { useState, useEffect, useCallback } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Paper, 
  List, 
  ListItem, 
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  TextField,
  Button,
  CircularProgress
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useDropzone } from 'react-dropzone';
import { Paper as PaperType } from './types';
import * as api from './api';

const App: React.FC = () => {
  const [papers, setPapers] = useState<PaperType[]>([]);
  const [question, setQuestion] = useState<string>('');
  const [answer, setAnswer] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [error, setError] = useState<string>('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadProgress(0);
    setError('');
    try {
      const result = await api.uploadPaper(file);
      fetchPapers();
      console.log('Upload result:', result);
      setUploadProgress(null);
    } catch (error) {
      console.error('Upload error:', error);
      setError('An error occurred while uploading the file.');
      setUploadProgress(null);
    }
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'application/pdf': ['.pdf']
    },
    onDrop
  });

  const fetchPapers = async () => {
    try {
      const response = await api.getPapers();
      setPapers(response.papers);
    } catch (error) {
      console.error('Error fetching papers:', error);
    }
  };

  const handleDeletePaper = async (paperId: string) => {
    try {
      await api.deletePaper(paperId);
      fetchPapers();
    } catch (error) {
      console.error('Error deleting paper:', error);
      setError('An error occurred while deleting the paper.');
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    try {
      const response = await api.askQuestion({ query: question });
      
      if (response && response.answer) {
        setAnswer(response.answer);
        setQuestion('');
      } else {
        setError('No answer was received.');
      }
    } catch (error: any) {
      setError('An error occurred while processing your question.');
      setAnswer('');
    }
    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (question.trim() && !loading) {
        handleAsk();
      }
    }
  };

  useEffect(() => {
    fetchPapers();
  }, []);

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Thesis Assistant
        </Typography>

        {/* File Upload Area */}
        <Paper
          {...getRootProps()}
          sx={{
            p: 3,
            mb: 3,
            textAlign: 'center',
            cursor: 'pointer',
            backgroundColor: '#f5f5f5'
          }}
        >
          <input {...getInputProps()} />
          <Typography>
            Drag and drop a PDF file or click to upload
          </Typography>
          {uploadProgress !== null && (
            <Box sx={{ mt: 2 }}>
              <CircularProgress variant="determinate" value={uploadProgress} />
              <Typography variant="caption" display="block">
                {uploadProgress}% uploaded
              </Typography>
            </Box>
          )}
        </Paper>

        {/* Paper List */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Uploaded Papers
          </Typography>
          <List>
            {papers.map((paper) => (
              <ListItem key={paper.id}>
                <ListItemText primary={paper.name} />
                <ListItemSecondaryAction>
                  <IconButton 
                    edge="end" 
                    aria-label="delete" 
                    onClick={() => handleDeletePaper(paper.id)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Paper>

        {/* Ask Question */}
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Ask a Question
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            placeholder="Ask a question about the paper..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            sx={{ mb: 2 }}
          />
          <Button
            variant="contained"
            onClick={handleAsk}
            disabled={loading || !question.trim()}
          >
            {loading ? <CircularProgress size={24} /> : 'Ask Question'}
          </Button>

          {error && (
            <Box sx={{ mt: 2, color: 'error.main' }}>
              <Typography>{error}</Typography>
            </Box>
          )}

          {answer && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6">Answer:</Typography>
              <Paper sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
                <Typography style={{ whiteSpace: 'pre-wrap' }}>{answer}</Typography>
              </Paper>
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
};

export default App;
