export const imageService = {
  validateImage: (file) => {
    const errors = [];
    
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      errors.push('Please upload a JPEG, PNG, or WebP image');
    }
    
    // Check file size (5MB limit)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
      errors.push('Image size should be less than 5MB');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  },

  createPreviewUrl: (file) => {
    return URL.createObjectURL(file);
  },

  revokePreviewUrl: (url) => {
    URL.revokeObjectURL(url);
  },

  compressImage: async (file, maxWidth = 800, quality = 0.8) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      
      img.onload = () => {
        const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;
        
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(resolve, file.type, quality);
      };
      
      img.src = URL.createObjectURL(file);
    });
  },


  getImageDimensions: (file) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        resolve({
          width: img.width,
          height: img.height
        });
      };
      img.src = URL.createObjectURL(file);
    });
  }
};