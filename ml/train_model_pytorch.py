
"""
Food Recognition Model Training Script - PyTorch Version
Uses Food-101 dataset to train a CNN model for food classification
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'dataset_path': './dataset',
    'model_save_path': './models',
    'img_size': 224,
    'batch_size': 8,  # Можна зменшити до 16 або 8 якщо не вистачає пам'яті
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,  # Windows compatibility
    'pin_memory': False
}

class Food101Dataset(Dataset):
    """Custom Dataset для Food-101"""
    
    def __init__(self, root_dir, transform=None, train=True, validation_split=0.2):
        self.root_dir = Path(root_dir) / 'images'
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Збираємо всі файли
        self.samples = []
        for class_name in self.classes:
            class_path = self.root_dir / class_name
            for img_path in class_path.glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        # Розділяємо на тренувальну та валідаційну вибірки
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(self.samples) * (1 - validation_split))
        
        if train:
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Повертаємо наступне зображення
            return self.__getitem__((idx + 1) % len(self.samples))

class FoodModelTrainer:
    def __init__(self, config):
        self.config = config
        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Створюємо директорії
        os.makedirs(config['model_save_path'], exist_ok=True)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def create_data_loaders(self):
        """Створює data loaders для тренування та валідації"""
        print("Creating data loaders...")
        
        # Трансформації для тренувальних даних (з аугментацією)
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(self.config['img_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Трансформації для валідаційних даних (без аугментації)
        val_transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Створюємо датасети
        train_dataset = Food101Dataset(
            self.config['dataset_path'], 
            transform=train_transform, 
            train=True,
            validation_split=self.config['validation_split']
        )
        
        val_dataset = Food101Dataset(
            self.config['dataset_path'], 
            transform=val_transform, 
            train=False,
            validation_split=self.config['validation_split']
        )
        
        # Зберігаємо назви класів
        self.class_names = train_dataset.classes
        with open(os.path.join(self.config['model_save_path'], 'class_names.json'), 'w') as f:
            json.dump(self.class_names, f, indent=2)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Number of classes: {len(self.class_names)}")
        
        # Створюємо data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
    
    def create_model(self):
        """Створює модель на основі ResNet50 з transfer learning"""
        print("Creating model...")
        
        # Використовуємо ResNet50 як базову модель
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Заморожуємо параметри базової моделі
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Замінюємо останній шар для нашої кількості класів
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
        
        # Розморожуємо останні кілька шарів для fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        self.model = self.model.to(self.device)
        
        # Ініціалізуємо оптимізатор та функцію втрат
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        # Scheduler для зменшення learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.config['reduce_lr_patience']
        )
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} trainable parameters")
    
    def train_epoch(self):
        """Тренування однієї епохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Оновлюємо progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Валідація однієї епохи"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, (data, targets) in enumerate(progress_bar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self):
        """Основний цикл тренування"""
        print("Starting training...")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Тренування
            train_loss, train_acc = self.train_epoch()
            
            # Валідація
            val_loss, val_acc = self.validate_epoch()
            
            # Зберігаємо історію
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping та збереження найкращої моделі
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Зберігає checkpoint моделі"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'class_names': self.class_names,
            'config': self.config
        }
        
        # Зберігаємо останній checkpoint
        torch.save(checkpoint, os.path.join(self.config['model_save_path'], 'latest_checkpoint.pth'))
        
        # Зберігаємо найкращий checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.config['model_save_path'], 'best_model.pth'))
    
    def save_model_for_inference(self):
        """Зберігає модель для inference"""
        print("Saving model for inference...")
        
        # Завантажуємо найкращу модель
        checkpoint = torch.load(os.path.join(self.config['model_save_path'], 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Зберігаємо тільки state_dict для inference
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'config': self.config
        }, os.path.join(self.config['model_save_path'], 'food_model_pytorch.pth'))
        
        # Зберігаємо конфігурацію
        config_data = {
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'img_size': self.config['img_size'],
            'model_type': 'resnet50'
        }
        
        with open(os.path.join(self.config['model_save_path'], 'model_config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print("Model saved successfully!")
    
    def plot_training_history(self):
        """Відображає графіки тренування"""
        if not self.history['train_loss']:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Training Loss', color='blue')
        axes[0].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Training Accuracy', color='blue')
        axes[1].plot(self.history['val_acc'], label='Validation Accuracy', color='red')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['model_save_path'], 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """Оцінює модель на валідаційних даних"""
        print("Evaluating model...")
        
        # Завантажуємо найкращу модель
        checkpoint = torch.load(os.path.join(self.config['model_save_path'], 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        val_loss, val_acc = self.validate_epoch()
        
        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        
        return val_loss, val_acc

def main():
    """Основна функція тренування"""
    print("Food Recognition Model Training - PyTorch Version")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    # Ініціалізуємо trainer
    trainer = FoodModelTrainer(CONFIG)
    
    try:
        # Створюємо data loaders
        trainer.create_data_loaders()
        
        # Створюємо модель
        trainer.create_model()
        
        # Тренуємо модель
        trainer.train_model()
        
        # Оцінюємо модель
        trainer.evaluate_model()
        
        # Зберігаємо модель для inference
        trainer.save_model_for_inference()
        
        # Відображаємо графіки
        trainer.plot_training_history()
        
        print("\nTraining completed successfully!")
        print(f"Models saved to: {CONFIG['model_save_path']}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    trainer = FoodModelTrainer(CONFIG)
    trainer.create_data_loaders()
    trainer.create_model()
    trainer.train_model()