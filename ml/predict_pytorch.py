#!/usr/bin/env python3
"""
Food Recognition Model Inference Script - PyTorch Version
Updated to handle file uploads from FastAPI
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import io
from typing import Union

class FoodPredictor:
    def __init__(self, model_path='./models'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.config = {}
        self.transform = None
        
        self.load_model()
        self.create_transform()
    
    def load_model(self):
        """Завантажує тренований PyTorch модель"""
        try:
            # Завантажуємо конфігурацію
            config_path = os.path.join(self.model_path, 'model_config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            self.class_names = self.config['class_names']
            
            # Створюємо модель
            self.model = models.resnet50(weights=None)
            
            # Замінюємо останній шар
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, len(self.class_names))
            )
            
            # Завантажуємо веса
            model_file = os.path.join(self.model_path, 'food_model_pytorch.pth')
            checkpoint = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Переводимо в режим inference
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully with {len(self.class_names)} classes")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def create_transform(self):
        """Створює трансформації для preprocessing"""
        img_size = self.config.get('img_size', 224)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image_from_bytes(self, image_bytes: bytes):
        """Preprocessing зображення з bytes для prediction"""
        try:
            # Створюємо BytesIO об'єкт з bytes
            image_stream = io.BytesIO(image_bytes)
            
            # Завантажуємо зображення з stream
            image = Image.open(image_stream).convert('RGB')
            
            # Застосовуємо трансформації
            image_tensor = self.transform(image)
            
            # Додаємо batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing image from bytes: {str(e)}")
            raise
    
    def preprocess_image_from_path(self, image_path: str):
        """Preprocessing зображення з файлу для prediction (для зворотної сумісності)"""
        try:
            # Завантажуємо зображення
            image = Image.open(image_path).convert('RGB')
            
            # Застосовуємо трансформації
            image_tensor = self.transform(image)
            
            # Додаємо batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing image from path: {str(e)}")
            raise
    
    def predict_from_bytes(self, image_bytes: bytes, top_k=5):
        """Передбачає клас їжі для зображення з bytes"""
        try:
            # Preprocessing
            image_tensor = self.preprocess_image_from_bytes(image_bytes)
            
            # Prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Отримуємо top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            top_predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                
                top_predictions.append({
                    'class': self.class_names[idx],
                    'confidence': float(prob),
                    'percentage': float(prob * 100)
                })
            
            return {
                'success': True,
                'predictions': top_predictions,
                'top_prediction': top_predictions[0]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, image_input: Union[str, bytes], top_k=5):
        """Універсальна функція prediction - приймає або шлях до файлу або bytes"""
        if isinstance(image_input, bytes):
            return self.predict_from_bytes(image_input, top_k)
        else:
            # Для зворотної сумісності - якщо передано шлях до файлу
            try:
                image_tensor = self.preprocess_image_from_path(image_input)
                
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                top_predictions = []
                for i in range(top_k):
                    idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    
                    top_predictions.append({
                        'class': self.class_names[idx],
                        'confidence': float(prob),
                        'percentage': float(prob * 100)
                    })
                
                return {
                    'success': True,
                    'predictions': top_predictions,
                    'top_prediction': top_predictions[0]
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
    
    def predict_batch(self, image_paths, top_k=5):
        """Передбачає класи для кількох зображень"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, top_k)
            results.append(result)
        return results

def main():
    """Основна функція для command line usage"""
    parser = argparse.ArgumentParser(description='Food Recognition Inference - PyTorch')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--model_path', default='./models', help='Path to model directory')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    try:
        # Ініціалізуємо predictor
        predictor = FoodPredictor(args.model_path)
        
        # Робимо prediction
        result = predictor.predict(args.image_path, args.top_k)
        
        # Виводимо результат
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(error_result, f, indent=2)
        else:
            print(json.dumps(error_result, indent=2))
        
        sys.exit(1)

if __name__ == "__main__":
    main()