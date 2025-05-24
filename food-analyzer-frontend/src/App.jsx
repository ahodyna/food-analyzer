import React from 'react';
import './styles/globals.css';
import './App.css';
import ImageUpload from './components/ImageUpload/ImageUpload';
import { useImageUpload } from './hooks/useImageUpload';
import { useFoodAnalysis } from './hooks/useFoodAnalysis';
import ResultsDisplay from './components/ResultsDisplay/ResultsDisplay';

function App() {
  const {
    selectedFile,
    previewUrl,
    error,
    isValidating,
    handleFileSelect,
    clearSelection
  } = useImageUpload();

  const {
    analysis,
    loading,
    error: analysisError,
    uploadProgress,
    analyzeFood,
    clearAnalysis
  } = useFoodAnalysis();

  const handleSend = async () => {
    if (!selectedFile) return alert('Please select a file first');
    try {
      await analyzeFood(selectedFile);
      alert('Image sent to mocked server!');
    } catch (e) {
      alert('Failed to send image: ' + e.message);
    }
  };

  return (
    <div className="App">
      <h1 className="page-title">Know Your Meal: Protein, Fat & Carbs</h1>
      <ImageUpload
        selectedFile={selectedFile}
        previewUrl={previewUrl}
        error={error}
        isValidating={isValidating}
        onFileSelect={handleFileSelect}
        onClear={clearSelection}
      />

      <div className="button-container">
        <button
          onClick={handleSend}
          disabled={loading || isValidating}
          className={`btn btn-primary ${loading ? 'loading' : ''}`}
        >
          {loading ? `Uploading... ${uploadProgress}%` : 'Send Image'}
        </button>
      </div>

      {analysis && (
        <ResultsDisplay
          analysis={analysis}
          onNewAnalysis={clearAnalysis}
        />
      )}

      {analysisError && <p style={{ color: 'red' }}>{analysisError}</p>}
    </div>
  );
}

export default App;
