export const formatNutritionValue = (value, unit = 'g') => {
  if (typeof value !== 'number') return '0g';
  return `${Math.round(value * 10) / 10}${unit}`;
};

export const calculateCalories = (protein, fat, carbs) => {
  return Math.round((protein * 4) + (fat * 9) + (carbs * 4));
};

export const getConfidenceColor = (confidence) => {
  if (confidence >= 0.8) return '#4CAF50'; // Green
  if (confidence >= 0.6) return '#FF9800'; // Orange
  return '#f44336'; // Red
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const capitalizeWords = (str) => {
  return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};