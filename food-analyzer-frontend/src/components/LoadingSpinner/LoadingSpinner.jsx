import React from 'react';
import './LoadingSpinner.css';

const LoadingSpinner = ({ 
  size = 'medium', 
  message = 'Loading...', 
  progress = null 
}) => {
  return (
    <div className={`loading-spinner ${size}`}>
      <div className="spinner-circle">
        <div className="spinner-inner"></div>
      </div>
      {message && <p className="loading-message">{message}</p>}
      {progress !== null && (
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <span className="progress-text">{progress}%</span>
        </div>
      )}
    </div>
  );
};

export default LoadingSpinner;