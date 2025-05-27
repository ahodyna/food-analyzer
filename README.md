# Food Analyzer - Deep Learning Project

This project allows you to analyze food from photographs, identify the dish, and get its nutritional information (calories, proteins, fats, carbohydrates).


### 🏗️ Machine Learning Architecture

 1. **ResNet50 - Residual Neural Network**
 2. **EfficientNet-B0 - Efficient Neural Architecture**
 3. **Vision Transformer (ViT) - Attention-Based Architecture**


### 📊 Model Performance & Evaluation

#### **Training Configuration**
| Model | Parameters | Input Size | Batch Size | Training Time | Memory Usage |
|-------|------------|------------|------------|---------------|--------------|
| ResNet50 | ~25M | 224×224 | 8 | ~4 hours | ~6GB GPU |
| EfficientNet-B0 | ~5M | 224×224 | 32 | ~2 hours | ~4GB GPU |
| ViT-Base | ~86M | 224×224 | 32 | ~6 hours | ~8GB GPU |


### 🎯 Dataset & Preprocessing

#### **Food-101 Dataset**
- **Scale**: 101 food categories, 1,000 images per class
- **Total Images**: 101,000 high-resolution food photographs
- **Split**: 75,750 training, 25,250 testing


## 🏛️ System Architecture

### **Three-Tier Architecture**

#### **1. Frontend (React.js)**
- **Image Upload**: Drag-and-drop functionality with real-time preview
- **Visualization**: Interactive charts for nutritional breakdown

#### **2. Backend API (Node.js + Express)**
- **File Processing**: Image upload and validation
- **AI Integration**: OpenAI GPT-4 for nutritional analysis

#### **3. ML Service (PyTorch + FastAPI)**
- **Model Serving**: Deep learning inference pipeline
- **Batch Processing**: Multiple image prediction support
- **Performance Optimization**: GPU acceleration and model caching
- **Scalability**: Horizontal scaling capabilities

### **Data Flow Architecture**
```
User Upload → Frontend Validation → Backend Processing → ML Inference → 
Nutritional Analysis (GPT-4) → Response Aggregation → Frontend Display
```

## 🔧 Technical Implementation

### **Environment Setup**
```bash
# ML Service Dependencies
pip install torch torchvision timm fastapi uvicorn pillow numpy

# Backend Dependencies  
npm install express multer axios openai cors dotenv

# Frontend Dependencies
npm install react framer-motion chart.js react-chartjs-2 axios
```

### **Model Training Commands**
```bash
# ResNet50 Training
python train_resnet.py --epochs 20 --batch_size 8 --lr 0.001

# EfficientNet Training  
python train_efficientnet.py --epochs 25 --batch_size 32

# Vision Transformer Training
python train_vit.py --model_name vit_base_patch16_224 --epochs 20
```

### **Deployment Configuration**
```bash
# ML Service (Port 5000)
uvicorn inference:app --host 0.0.0.0 --port 5000

# Backend API (Port 3001)
npm start

# Frontend (Port 3000)
npm run build && serve -s build
```