const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

router.post('/analyze-food', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file uploaded' });
        }

        const formData = new FormData();
        formData.append('image', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype,
            knownLength: req.file.size
        });


        // const mlResponse = await axios.post('http://ml-model:5000/predict', formData, {
        //   headers: formData.getHeaders()
        // });

        const mlResponse = {
            food_type: 'pizza',
            confidence: 0.85,
            nutrition: {
                protein: 10,
                fat: 12,
                carbohydrates: 25,
            }
        };

        res.json(mlResponse);
    } catch (error) {
        console.error('Error during food analysis:', error.message);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

module.exports = router;
