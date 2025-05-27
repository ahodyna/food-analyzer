const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const router = express.Router();
const openai = require('../utils/openaiClient');
const { extractJson } = require('../utils/parseFile')

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB максимум
    },
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'), false);
        }
    }
});

router.post('/analyze-food', upload.single('image'), async (req, res) => {
    try {
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

        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const topK = req.body.top_k || req.query.top_k || 5;

        const response = await axios.post(`http://localhost:5000/predict?top_k=${topK}`, formData, {
            headers: {
                ...formData.getHeaders(),
                'Accept': 'application/json'
            },
            timeout: 30000,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        console.log('ML Response received:', {
            success: response.data.success,
            top_prediction: response.data.top_prediction?.class,
            confidence: response.data.top_prediction?.confidence
        });

        const predictedClass = response.data.top_prediction?.class;

        let nutritionInfo = null;
        if (predictedClass) {
            try {
                const completion = await openai.chat.completions.create({
                    model: "gpt-4",
                    messages: [
                        { role: "system", content: "You are a nutrition expert." },
                        {
                            role: "user", content: `Give me approximate calories, protein, fat, and carbohydrates per 100g of ${predictedClass}. Respond in valid JSON format without: 
                        {
                        "calories": "value", "protein": "value", "fat": "value", "carbohydrates": "value"
                        }` }
                    ],
                    temperature: 0.2
                });


                console.log('message', completion.choices[0].message.content)
                const message = completion.choices[0].message.content;
                nutritionInfo = extractJson(message);

            } catch (e) {
                console.error("OpenAI Nutrition Query Error:", e.message);
            }
        }


        const isLowAccuracy = response.data.top_prediction?.confidence < 0.7;

        res.json({
            success: true,
            food_type: predictedClass,
            confidence: response.data.top_prediction?.confidence,
            nutrition: nutritionInfo,
            message: isLowAccuracy ? 'Low accuracy warning' : 'Food analysis completed successfully',
            warnings: {
                lowAccuracy: isLowAccuracy,
                noNutrition: !nutritionInfo
            }
        });



    } catch (error) {
        console.error('Error during food analysis:', error.message);

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
            return res.status(error.response.status || 500).json({
                success: false,
                error: error.response.data?.detail || 'Error from ML service',
                ml_error: error.response.data
            });
        } else if (error.code === 'ECONNREFUSED') {
            return res.status(503).json({
                success: false,
                error: 'ML service is currently unavailable'
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


module.exports = router;
