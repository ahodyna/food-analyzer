import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import './ImageUpload.css';

const ImageUpload = ({ 
  onFileSelect, 
  selectedFile, 
  previewUrl, 
  error, 
  isValidating,
  onClear 
}) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    multiple: false,
    maxSize: 5 * 1024 * 1024 // 5MB
  });

  return (
    <div className="image-upload-container">
      {!previewUrl ? (
        <motion.div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''} ${error ? 'error' : ''}`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <input {...getInputProps()} />
          <div className="dropzone-content">
            <div className="upload-icon">📸</div>
            <h3>Upload Food Image</h3>
            <p>
              {isDragActive
                ? 'Drop the image here...'
                : 'Drag & drop an image here, or click to select'
              }
            </p>
            <div className="file-requirements">
              <small>Supports: JPEG, PNG, WebP (max 5MB)</small>
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div 
          className="image-preview"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <img src={previewUrl} alt="Food preview" />
          <div className="image-overlay">
            <button 
              className="btn btn-secondary clear-btn"
              onClick={onClear}
            >
              ✕ Clear
            </button>
          </div>
          <div className="image-info">
            <span className="file-name">{selectedFile?.name}</span>
            <span className="file-size">
              {selectedFile && (selectedFile.size / 1024).toFixed(1)} KB
            </span>
          </div>
        </motion.div>
      )}
      
      {error && (
        <motion.div 
          className="error-message"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {error}
        </motion.div>
      )}
    </div>
  );
};

export default ImageUpload;