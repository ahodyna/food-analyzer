import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

// Create axios instance with default config
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 seconds for image processing
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor
apiClient.interceptors.request.use(
    (config) => {
        // Add auth token if needed
        const token = localStorage.getItem('authToken');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        const message = error.response?.data?.message || error.message || 'Something went wrong';
        return Promise.reject(new Error(message));
    }
);

// API functions
export const foodAnalysisAPI = {
      analyzeFood: async (imageFile, options = {}) => {

        const formData = new FormData();
        formData.append('image', imageFile);

        // Add any additional options
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

    // analyzeFood: async (imageFile, options = {}) => {
    //     // Simulate network delay
    //     await new Promise(resolve => setTimeout(resolve, 1000));
    //     // Simulate upload progress
    //     if (options.onUploadProgress) {
    //         for (let p = 0; p <= 100; p += 20) {
    //             options.onUploadProgress({ loaded: p, total: 100 });
    //             await new Promise(resolve => setTimeout(resolve, 100));
    //         }
    //     }
    //     // Return mocked response
    //     return {
    //         food_type: 'pizza',
    //         confidence: 0.85,
    //         nutrition: {
    //             protein: 10,
    //             fat: 12,
    //             carbohydrates: 25,
    //         }
    //     };
    // },

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