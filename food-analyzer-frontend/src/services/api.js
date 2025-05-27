import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 seconds for image processing
    headers: {
        'Content-Type': 'application/json',
    },
});

apiClient.interceptors.request.use(
    (config) => {
    
        const token = localStorage.getItem('authToken');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        const message = error.response?.data?.message || error.message || 'Something went wrong';
        return Promise.reject(new Error(message));
    }
);

export const foodAnalysisAPI = {
      analyzeFood: async (imageFile, options = {}) => {

        const formData = new FormData();
        formData.append('image', imageFile);

        Object.keys(options).forEach(key => {
          formData.append(key, options[key]);
        });

        const response = await apiClient.post('/analyze-food', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: options.onUploadProgress,
        });

        return response.data;
      },

    getAnalysisHistory: async () => {
        const response = await apiClient.get('/analysis-history');
        return response.data;
    },

    getFoodDatabase: async () => {
        const response = await apiClient.get('/food-database');
        return response.data;
    },
};

export default apiClient;