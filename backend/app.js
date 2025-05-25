const express = require('express');
const cors = require('cors');
const analyzeRoute = require('./routes/analyze');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/', analyzeRoute);

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Backend server running on port ${PORT}`);
});
