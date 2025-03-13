import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [response, setResponse] = useState<{ confidence: number; predicted_class: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setImagePreview(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }

      const result = await response.json();
      setResponse(result);
      setError(null);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message);
      setResponse(null);
    }
  };

  return (
    <div className="App">
      <h1>Battery Type Classifier</h1>
      <p>Upload an image of a battery to classify its type.</p>
      <label className="label-file-input" htmlFor="file-input">Choose File</label>
      <input id="file-input" type="file" accept="image/*" onChange={handleFileChange} />
      <div className="image-preview-container">
        {imagePreview ? (
          <img src={imagePreview} alt="Selected" />
        ) : (
          <div className="image-placeholder">No image selected</div>
        )}
      </div>
      <button onClick={handleSubmit} disabled={!selectedFile}>
        Submit
      </button>
      {error && (
        <div className="error-container">
          <h2>Error</h2>
          <p>{error}</p>
        </div>
      )}
      {response && (
        <div className="response-container">
          <h2>Prediction Result</h2>
          <h4>Predicted Class: {response.predicted_class}</h4>
          <h4>Confidence: {response.confidence.toFixed(2)}</h4>
        </div>
      )}
    </div>
  );
}

export default App;
