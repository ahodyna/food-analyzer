import { useState, useCallback } from 'react';
import { imageService } from '../services/imageService';

export const useImageUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState(null);
  const [isValidating, setIsValidating] = useState(false);

  const handleFileSelect = useCallback(async (files) => {
    const file = files[0];
    if (!file) return;

    setIsValidating(true);
    setError(null);

    try {
      const validation = imageService.validateImage(file);
      if (!validation.isValid) {
        setError(validation.errors.join(', '));
        return;
      }
      const dimensions = await imageService.getImageDimensions(file);
      
      const processedFile = file.size > 1024 * 1024 
        ? await imageService.compressImage(file)
        : file;

      const preview = imageService.createPreviewUrl(processedFile);

      if (previewUrl) {
        imageService.revokePreviewUrl(previewUrl);
      }

      setSelectedFile(processedFile);
      setPreviewUrl(preview);
      
    } catch (err) {
      setError('Failed to process image');
    } finally {
      setIsValidating(false);
    }
  }, [previewUrl]);

  const clearSelection = useCallback(() => {
    if (previewUrl) {
      imageService.revokePreviewUrl(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
  }, [previewUrl]);

  return {
    selectedFile,
    previewUrl,
    error,
    isValidating,
    handleFileSelect,
    clearSelection
  };
};