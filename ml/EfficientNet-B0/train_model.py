import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Food101Dataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                split_idx = int(0.8 * len(images))
                if train:
                    selected_images = images[:split_idx]
                else:
                    selected_images = images[split_idx:]
                
                for img_name in selected_images:
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Завантажено {'тренувальних' if train else 'валідаційних'} зображень: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Food101Dataset('dataset/images', transform=train_transform, train=True)
val_dataset = Food101Dataset('dataset/images', transform=val_transform, train=False)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

if __name__ == '__main__':

    class Food101Model(nn.Module):
        def __init__(self, num_classes):
            super(Food101Model, self).__init__()
            # Завантажуємо предтреновану EfficientNet-B0
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)

    num_classes = len(train_dataset.classes)
    model = Food101Model(num_classes).to(device)

    print(f"Модель створена для {num_classes} класів")
    print(f"Кількість параметрів: {sum(p.numel() for p in model.parameters()):,}")

    # Налаштування оптимізатора та функції втрат
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def train_epoch(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Тренування")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/len(pbar):.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        return running_loss / len(train_loader), 100 * correct / total

    def validate_epoch(model, val_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Валідація")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/len(pbar):.4f}',
                    'Acc': f'{100*correct/total:.2f}%'
                })
        
        return running_loss / len(val_loader), 100 * correct / total

    num_epochs = 25
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Починаємо тренування...")
    for epoch in range(num_epochs):
        print(f"\nЕпоха {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
 
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Тренувальна втрата: {train_loss:.4f}, Точність: {train_acc:.2f}%")
        print(f"Валідаційна втрата: {val_loss:.4f}, Точність: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes
            }, 'best_food101_model.pth')
            print(f"✓ Збережено нову найкращу модель з точністю {val_acc:.2f}%")

    print(f"\nТренування завершено! Найкраща точність валідації: {best_val_acc:.2f}%")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Тренувальна втрата')
    plt.plot(val_losses, label='Валідаційна втрата')
    plt.title('Втрата під час тренування')
    plt.xlabel('Епоха')
    plt.ylabel('Втрата')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Тренувальна точність')
    plt.plot(val_accs, label='Валідаційна точність')
    plt.title('Точність під час тренування')
    plt.xlabel('Епоха')
    plt.ylabel('Точність (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(np.array(train_accs) - np.array(val_accs))
    plt.title('Різниця точності (Overfit)')
    plt.xlabel('Епоха')
    plt.ylabel('Тренувальна - Валідаційна (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    def predict_image(model, image_path, classes, transform, device):
        model.eval()
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return classes[predicted.item()], confidence.item()

    print("\nПриклад передбачення:")
    print("predicted_class, confidence = predict_image(model, 'path/to/image.jpg', train_dataset.classes, val_transform, device)")
    print("print(f'Передбачений клас: {predicted_class}, Впевненість: {confidence:.3f}')")

    with open('food101_classes.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset.classes, f, ensure_ascii=False, indent=2)