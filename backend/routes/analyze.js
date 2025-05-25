const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const router = express.Router();

// Конфігурація multer для обробки файлів
const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB максимум
    },
    fileFilter: (req, file, cb) => {
        // Перевіряємо чи це зображення
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'), false);
        }
    }
});

// Маршрут для аналізу їжі
router.post('/analyze-food', upload.single('image'), async (req, res) => {
    try {
        // Перевіряємо чи файл завантажився
        if (!req.file) {
            return res.status(400).json({ 
                error: 'No image file uploaded',
                success: false 
            });
        }

        console.log('Processing file:', {
            originalname: req.file.originalname,
            mimetype: req.file.mimetype,
            size: req.file.size
        });

        // Створюємо FormData для відправки на FastAPI
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        // Параметри для prediction
        const topK = req.body.top_k || req.query.top_k || 5;

        // Відправляємо запит на FastAPI сервер
        const response = await axios.post(`http://localhost:5000/predict?top_k=${topK}`, formData, {
            headers: {
                ...formData.getHeaders(),
                'Accept': 'application/json'
            },
            timeout: 30000, // 30 секунд timeout
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        console.log('ML Response received:', {
            success: response.data.success,
            top_prediction: response.data.top_prediction?.class,
            confidence: response.data.top_prediction?.confidence
        });

        // Повертаємо результат
        res.json({
            success: true,
            data: response.data,
            message: 'Food analysis completed successfully'
        });

    } catch (error) {
        console.error('Error during food analysis:', error.message);
        
        // Обробляємо різні типи помилок
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                success: false,
                error: 'File too large. Maximum size is 10MB'
            });
        }
        
        if (error.message === 'Only image files are allowed') {
            return res.status(400).json({
                success: false,
                error: 'Only image files are allowed'
            });
        }

        if (error.response) {
            // Помилка від FastAPI сервера
            return res.status(error.response.status || 500).json({
                success: false,
                error: error.response.data?.detail || 'Error from ML service',
                ml_error: error.response.data
            });
        } else if (error.code === 'ECONNREFUSED') {
            // ML сервер недоступний
            return res.status(503).json({
                success: false,
                error: 'ML service is currently unavailable'
            });
        } else {
            // Інші помилки
            return res.status(500).json({
                success: false,
                error: 'Internal Server Error',
                details: process.env.NODE_ENV === 'development' ? error.message : undefined
            });
        }
    }
});

// Маршрут для batch аналізу (кілька зображень)
router.post('/analyze-food-batch', upload.array('images', 10), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ 
                error: 'No image files uploaded',
                success: false 
            });
        }

        console.log(`Processing ${req.files.length} files`);

        const formData = new FormData();
        
        // Додаємо всі файли до FormData
        req.files.forEach((file, index) => {
            formData.append('files', file.buffer, {
                filename: file.originalname,
                contentType: file.mimetype
            });
        });

        const topK = req.body.top_k || req.query.top_k || 5;

        const response = await axios.post(`http://localhost:5000/predict-batch?top_k=${topK}`, formData, {
            headers: {
                ...formData.getHeaders(),
                'Accept': 'application/json'
            },
            timeout: 60000, // 60 секунд для batch processing
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        console.log(`Batch ML Response received for ${req.files.length} files`);

        res.json({
            success: true,
            data: response.data,
            message: `Batch food analysis completed for ${req.files.length} images`
        });

    } catch (error) {
        console.error('Error during batch food analysis:', error.message);
        
        if (error.response) {
            return res.status(error.response.status || 500).json({
                success: false,
                error: error.response.data?.detail || 'Error from ML service',
                ml_error: error.response.data
            });
        } else {
            return res.status(500).json({
                success: false,
                error: 'Internal Server Error',
                details: process.env.NODE_ENV === 'development' ? error.message : undefined
            });
        }
    }
});

// Health check endpoint
router.get('/ml-health', async (req, res) => {
    try {
        const response = await axios.get('http://localhost:5000/health', {
            timeout: 5000
        });
        
        res.json({
            success: true,
            ml_service: response.data
        });
    } catch (error) {
        res.status(503).json({
            success: false,
            error: 'ML service is not available',
            details: error.message
        });
    }
});

module.exports = router;