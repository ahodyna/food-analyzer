import { useState, useCallback } from 'react';
import { foodAnalysisAPI } from '../services/api';

export const useFoodAnalysis = () => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const analyzeFood = useCallback(async (imageFile, options = {}) => {
    if (!imageFile) {
      setError('No image file provided');
      return;
    }

    setLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const result = await foodAnalysisAPI.analyzeFood(imageFile, {
        ...options,
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });
      if (result.nutrition) {
        const nutrition = result.nutrition;
        const fixedNutrition = {
          calories: parseFloat(nutrition.calories),
          protein: parseFloat(nutrition.protein),
          fat: parseFloat(nutrition.fat),
          carbohydrates: parseFloat(nutrition.carbohydrates),
        };
        result.nutrition = fixedNutrition;
      }


      setAnalysis(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  }, []);

  const clearAnalysis = useCallback(() => {
    setAnalysis(null);
    setError(null);
    setUploadProgress(0);
  }, []);

  return {
    analysis,
    loading,
    error,
    uploadProgress,
    analyzeFood,
    clearAnalysis
  };
};