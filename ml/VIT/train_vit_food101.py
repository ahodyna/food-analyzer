"""
ViT Food-101 Training Script
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Food101Dataset(Dataset):
    
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label_idx = self.class_to_idx[label]
            return image, label_idx
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
    
            default_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                default_image = self.transform(default_image)
            return default_image, self.class_to_idx[label]

class Food101DataLoader:
    
    def __init__(self, dataset_path="dataset/images"):
        self.dataset_path = Path(dataset_path)
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def load_data(self, test_size=0.2, random_state=42):
        logger.info(f"Loading data from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path {self.dataset_path} does not exist!")

        class_folders = [d for d in self.dataset_path.iterdir() 
                        if d.is_dir() and not d.name.startswith('.')]
        
        if len(class_folders) == 0:
            raise ValueError(f"No class folders found in {self.dataset_path}")
        
        class_folders.sort()
        self.class_names = [folder.name for folder in class_folders]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        logger.info(f"Found {len(self.class_names)} classes")
        
        all_image_paths = []
        all_labels = []
        
        for class_folder in class_folders:
            class_name = class_folder.name
    
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_folder.glob(ext))
            
            for image_path in image_files:
                all_image_paths.append(str(image_path))
                all_labels.append(class_name)
        
        logger.info(f"Total images found: {len(all_image_paths)}")
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_image_paths, all_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=all_labels  # Рівномірний розподіл класів
        )
        
        logger.info(f"Train set: {len(train_paths)} images")
        logger.info(f"Validation set: {len(val_paths)} images")
        
        return (train_paths, train_labels), (val_paths, val_labels)

class ViTFoodTrainer:
    
    def __init__(self, model_name='vit_base_patch16_224', num_classes=101, device=None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Using device: {self.device}")
        
    def build_model(self, pretrained=True, freeze_backbone=False):
        logger.info(f"Building {self.model_name} model")
        
        self.model = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=self.num_classes
        )
        
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'head' not in name and 'classifier' not in name:
                    param.requires_grad = False
            logger.info("Backbone frozen, only head will be trained")
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def setup_training(self, learning_rate=1e-4, weight_decay=1e-4):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
    @staticmethod
    def get_transforms(img_size=224):
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def train_epoch(self, train_loader):

        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                current_acc = 100 * correct_predictions / total_samples
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=20, save_path="./models"):
        logger.info(f"Starting training for {epochs} epochs")
        
        os.makedirs(save_path, exist_ok=True)
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
         
            train_loss, train_acc = self.train_epoch(train_loader)
            
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_path, epoch, val_acc, is_best=True)
                logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
 
            if (epoch + 1) % 10 == 0:
                self.save_model(save_path, epoch, val_acc, is_best=False)
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def save_model(self, save_path, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(save_path, 'vit_food_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(save_path, f'vit_food_checkpoint_epoch_{epoch+1}.pth'))
    
    def plot_training_history(self, save_path="./models"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
 
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train ViT on Food-101')
    parser.add_argument('--dataset_path', default='dataset/images', help='Path to Food-101 images')
    parser.add_argument('--model_name', default='vit_base_patch16_224', help='ViT model variant')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--save_path', default='./models', help='Path to save models')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for Windows)')
    
    args = parser.parse_args()
    
    data_loader = Food101DataLoader(args.dataset_path)
    (train_paths, train_labels), (val_paths, val_labels) = data_loader.load_data()
    
    train_transform, val_transform = ViTFoodTrainer.get_transforms(args.img_size)
    
    train_dataset = Food101Dataset(train_paths, train_labels, data_loader.class_to_idx, train_transform)
    val_dataset = Food101Dataset(val_paths, val_labels, data_loader.class_to_idx, val_transform)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=use_pin_memory
    )
    

    os.makedirs(args.save_path, exist_ok=True)
    config = {
        'class_names': data_loader.class_names,
        'class_to_idx': data_loader.class_to_idx,
        'idx_to_class': data_loader.idx_to_class,
        'num_classes': len(data_loader.class_names),
        'img_size': args.img_size,
        'model_name': args.model_name
    }
    
    with open(os.path.join(args.save_path, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
  
    trainer = ViTFoodTrainer(
        model_name=args.model_name,
        num_classes=len(data_loader.class_names)
    )

    trainer.build_model(pretrained=True, freeze_backbone=args.freeze_backbone)
    trainer.setup_training(learning_rate=args.learning_rate)
    
    best_acc = trainer.train(train_loader, val_loader, args.epochs, args.save_path)
    
    trainer.plot_training_history(args.save_path)
    
    logger.info(f"Training completed with best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()