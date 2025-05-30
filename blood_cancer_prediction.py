# %% [markdown]
# # Blood Cancer Prediction Model
# 
# This notebook implements a deep learning model for blood cancer prediction using both image and genomic data. The model combines computer vision techniques with genomic analysis to provide accurate predictions.

# %% [markdown]
# ## Setup and Installation
# 
# First, let's clone the repository and install the required dependencies.

# %%
# Clone the repository
!git clone https://github.com/psg0009/CancerPredictionModel.git
%cd CancerPredictionModel

# %%
# Install required packages
!pip install torch torchvision torchaudio
!pip install opencv-python
!pip install scikit-learn
!pip install albumentations
!pip install imbalanced-learn
!pip install matplotlib
!pip install pandas
!pip install numpy
!pip install tqdm

# %% [markdown]
# ## Import Required Libraries

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import albumentations as A
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pandas as pd

# %% [markdown]
# ## Set Random Seeds for Reproducibility

# %%
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# %% [markdown]
# ## Data Generation and Processing
# 
# Let's create synthetic data for our blood cancer prediction model.

# %%
class BloodCancerDataset(Dataset):
    def __init__(self, images, masks, genomic_data, labels, transform=None):
        self.images = images
        self.masks = masks
        self.genomic_data = genomic_data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        genomic = self.genomic_data[idx]
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return {
            'image': torch.FloatTensor(image),
            'mask': torch.FloatTensor(mask),
            'genomic': torch.FloatTensor(genomic),
            'label': torch.LongTensor([label])[0]
        }

# %%
def generate_synthetic_data(n_samples=1000, image_size=(224, 224), n_genomic_features=100):
    # Generate synthetic images
    images = np.random.rand(n_samples, 3, *image_size)
    masks = np.random.rand(n_samples, 1, *image_size)
    
    # Generate synthetic genomic data
    genomic_data = np.random.rand(n_samples, n_genomic_features)
    
    # Generate synthetic labels (0: normal, 1: cancer)
    labels = np.random.randint(0, 2, n_samples)
    
    return images, masks, genomic_data, labels

# %% [markdown]
# ## Model Architecture

# %%
class BloodCancerAnalyzer(nn.Module):
    def __init__(self, n_genomic_features=100):
        super(BloodCancerAnalyzer, self).__init__()
        
        # Image processing branch
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Genomic data processing branch
        self.genomic_encoder = nn.Sequential(
            nn.Linear(n_genomic_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, image, genomic):
        # Process image
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Process genomic data
        genomic_features = self.genomic_encoder(genomic)
        
        # Combine features
        combined = torch.cat([image_features, genomic_features], dim=1)
        
        # Classification
        output = self.classifier(combined)
        return output

# %% [markdown]
# ## Training Setup

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = batch['image'].to(device)
            genomic = batch['genomic'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, genomic)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                genomic = batch['genomic'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, genomic)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')

# %% [markdown]
# ## Main Training Loop

# %%
# Generate synthetic data
images, masks, genomic_data, labels = generate_synthetic_data(n_samples=1000)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    list(range(len(images))), labels, test_size=0.2, random_state=42, stratify=labels
)

# Define transforms
train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
    ], p=0.3),
    A.OneOf([
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
    ], p=0.3),
])

# Create datasets
train_dataset = BloodCancerDataset(
    images=[images[i] for i in X_train],
    masks=[masks[i] for i in X_train],
    genomic_data=[genomic_data[i] for i in X_train],
    labels=[labels[i] for i in X_train],
    transform=train_transform
)

val_dataset = BloodCancerDataset(
    images=[images[i] for i in X_val],
    masks=[masks[i] for i in X_val],
    genomic_data=[genomic_data[i] for i in X_val],
    labels=[labels[i] for i in X_val]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model and training components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BloodCancerAnalyzer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)

# %% [markdown]
# ## Visualization and Analysis

# %%
def plot_training_curves(train_losses, val_losses, train_f1s, val_f1s):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show() 