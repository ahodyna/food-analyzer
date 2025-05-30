export const validateImageFile = (file) => {
  const errors = [];
  
  if (!file) {
    errors.push('Please select an image file');
    return { isValid: false, errors };
  }
  
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
  if (!allowedTypes.includes(file.type.toLowerCase())) {
    errors.push('Please upload a JPEG, PNG, or WebP image');
  }
  
  const maxSize = 5 * 1024 * 1024;
  if (file.size > maxSize) {
    errors.push('Image size should be less than 5MB');
  }
  
  if (!file.type.startsWith('image/')) {
    errors.push('Please upload a valid image file');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

export const validateAnalysisResult = (result) => {
  if (!result) return false;
  
  const requiredFields = ['food_type', 'confidence', 'nutrition'];
  const nutritionFields = ['protein', 'fat', 'carbohydrates'];
  
  for (const field of requiredFields) {
    if (!(field in result)) return false;
  }
  
  if (typeof result.nutrition !== 'object') return false;
  for (const field of nutritionFields) {
    if (!(field in result.nutrition)) return false;
    if (typeof result.nutrition[field] !== 'number') return false;
  }
  
  return true;
};