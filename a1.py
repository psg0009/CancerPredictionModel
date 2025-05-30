import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
from visualization_utils import (plot_wbc_samples, plot_training_history,
                               plot_genomic_features, plot_roc_curve,
                               plot_confusion_matrix, visualize_wbc_types,
                               save_visualization)
from sklearn.metrics import average_precision_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from collections import Counter
import shutil
from training_utils import (
    train_epoch,
    validate_epoch,
    generate_epoch_visualizations,
    generate_comparative_analysis
)

# ================================
# 🔹 Step 1: Setup Configuration
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Class balance configurations
BALANCE_RATIOS = [
    (50, 50),   # Balanced
    (55, 45),   # Slightly imbalanced
    (60, 40),
    (65, 35),
    (70, 30),
    (75, 25),
    (80, 20),
    (85, 15),
    (90, 10),
    (95, 5),
    (97, 3),
    (99, 1)     # Extremely imbalanced
]

# Imbalance handling strategies
IMBALANCE_STRATEGIES = [
    'weighted',      # Weighted loss and sampling
    'smote',        # SMOTE oversampling
    'tomek',        # Tomek links undersampling
    'hybrid',       # SMOTE + Tomek links
    'focal_loss'    # Focal Loss
]

# Base configuration
BASE_SAMPLES = 1000  # Total samples for each ratio
image_base_path = "./synthetic_cancer_images"

# WBC types and their characteristics
WBC_TYPES = {
    'lymphocyte': {'size': (8, 12), 'color': (100, 130, 255)},
    'neutrophil': {'size': (10, 15), 'color': (130, 100, 255)},
    'monocyte': {'size': (12, 18), 'color': (150, 100, 200)},
    'eosinophil': {'size': (10, 14), 'color': (100, 150, 255)},
    'basophil': {'size': (8, 12), 'color': (120, 120, 255)}
}

# Blood cancer specific genes
BLOOD_CANCER_GENES = [
    'FLT3', 'NPM1', 'DNMT3A', 'IDH1', 'IDH2', 'TET2', 'ASXL1', 'RUNX1',
    'TP53', 'CEBPA', 'WT1', 'NRAS', 'KIT', 'KRAS', 'JAK2', 'MPL'
]

def generate_wbc_image(size=128, is_cancerous=False):
    # Create background (blood plasma)
    img = np.ones((size, size, 3), dtype=np.uint8) * 245
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Add RBCs (background cells)
    for _ in range(50):
        x, y = np.random.randint(10, size-10, 2)
        radius = np.random.randint(3, 6)
        color = (130, 50, 50)  # RBC color
        cv2.circle(img, (x, y), radius, color, -1)
    
    # Add WBCs
    num_wbcs = np.random.randint(3, 8) if is_cancerous else np.random.randint(1, 4)
    for _ in range(num_wbcs):
        wbc_type = random.choice(list(WBC_TYPES.keys()))
        props = WBC_TYPES[wbc_type]
        
        x, y = np.random.randint(20, size-20, 2)
        radius = np.random.randint(*props['size'])
        
        # Modify appearance for cancerous cells
        if is_cancerous:
            # Add irregular shape and darker color for cancerous cells
            points = np.random.randint(-3, 4, (8, 2)) + np.array([[x, y]])
            color = tuple(max(0, c - 50) for c in props['color'])
            cv2.fillPoly(img, [points], color)
            cv2.fillPoly(mask, [points], 255)
        else:
            cv2.circle(img, (x, y), radius, props['color'], -1)
            cv2.circle(mask, (x, y), radius, 255, -1)
    
    return img, mask

def generate_blood_genomic_data(n_samples, is_cancerous):
    data = {}
    for gene in BLOOD_CANCER_GENES:
        if is_cancerous:
            # Simulate gene mutations/expressions in cancer
            data[gene] = np.random.normal(2.0, 0.5, n_samples)
        else:
            # Simulate normal gene expressions
            data[gene] = np.random.normal(1.0, 0.3, n_samples)
    return pd.DataFrame(data)

def generate_dataset_for_ratio(pos_ratio, neg_ratio, total_samples, output_dir):
    """Generate synthetic dataset for a specific class balance ratio."""
    # Clean output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of samples for each class
    n_positive = int(total_samples * pos_ratio / 100)
    n_negative = total_samples - n_positive
    
    print(f"Generating dataset with ratio {pos_ratio}:{neg_ratio}")
    print(f"Positive samples: {n_positive}, Negative samples: {n_negative}")
    
    # Generate images and masks
    image_filenames = []
    
    # Generate positive (cancerous) samples
    for i in range(n_positive):
        img, mask = generate_wbc_image(is_cancerous=True)
        img_path = os.path.join(output_dir, f"cancerous_{i}.png")
        mask_path = os.path.join(output_dir, f"cancerous_{i}_mask.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
        image_filenames.append(img_path)
    
    # Generate negative (non-cancerous) samples
    for i in range(n_negative):
        img, mask = generate_wbc_image(is_cancerous=False)
        img_path = os.path.join(output_dir, f"non_cancerous_{i}.png")
        mask_path = os.path.join(output_dir, f"non_cancerous_{i}_mask.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask)
        image_filenames.append(img_path)
    
    # Generate genomic data
    pos_genomic = generate_blood_genomic_data(n_positive, is_cancerous=True)
    neg_genomic = generate_blood_genomic_data(n_negative, is_cancerous=False)
    
    genomic_data = pd.concat([pos_genomic, neg_genomic])
    genomic_data["Label"] = [1] * n_positive + [0] * n_negative
    
    # Save genomic data
    genomic_data.to_csv(os.path.join(output_dir, "genomic_data.csv"), index=False)
    
    return image_filenames, genomic_data

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def apply_smote(X, y):
    """Apply SMOTE oversampling."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def apply_tomek(X, y):
    """Apply Tomek links undersampling."""
    tomek = TomekLinks()
    X_res, y_res = tomek.fit_resample(X, y)
    return X_res, y_res

def apply_hybrid_sampling(X, y):
    """Apply SMOTE followed by Tomek links."""
    X_smote, y_smote = apply_smote(X, y)
    X_res, y_res = apply_tomek(X_smote, y_smote)
    return X_res, y_res

def get_imbalance_handler(strategy, pos_ratio, neg_ratio):
    """Get imbalance handling components based on strategy."""
    if strategy == 'weighted':
        class_weights = torch.FloatTensor([neg_ratio/100, pos_ratio/100])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_ratio/neg_ratio]))
        sampler = None
    elif strategy == 'focal_loss':
        class_weights = None
        criterion = FocalLoss(alpha=neg_ratio/100)
        sampler = None
    else:
        class_weights = None
        criterion = nn.BCEWithLogitsLoss()
        sampler = None  # Sampling will be handled in the dataset
    
    return criterion, class_weights, sampler

class ImbalancedCancerDataset(Dataset):
    def __init__(self, image_filenames, genomic_data, labels, transform=None, 
                 strategy='weighted', is_train=True):
        self.image_filenames = image_filenames
        self.labels = labels.values.astype(np.float32)
        self.transform = transform
        self.is_train = is_train
        self.strategy = strategy
        self.mask_filenames = [f.replace('.png', '_mask.png') for f in image_filenames]
        
        # Apply sampling strategies if training
        if is_train and strategy in ['smote', 'tomek', 'hybrid']:
            if strategy == 'smote':
                genomic_data, self.labels = apply_smote(genomic_data, self.labels)
            elif strategy == 'tomek':
                genomic_data, self.labels = apply_tomek(genomic_data, self.labels)
            else:  # hybrid
                genomic_data, self.labels = apply_hybrid_sampling(genomic_data, self.labels)
        
        self.genomic_data = genomic_data.values.astype(np.float32)
        
        print(f"Dataset distribution after {strategy}: {Counter(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_filenames[idx])
        if img is None:
            raise ValueError(f"Could not read image file: {self.image_filenames[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask with fallback to empty mask if file is missing
        mask_path = self.mask_filenames[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Create an empty mask with the same dimensions as the image
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            print(f"Warning: Mask file not found: {mask_path}, using empty mask")
        
        # Apply transformations if specified
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # Get genomic features and label
        genomic_features = self.genomic_data[idx]
        label = self.labels[idx]
        
        return img, mask, genomic_features, label

class BloodCancerAnalyzer(nn.Module):
    def __init__(self):
        super(BloodCancerAnalyzer, self).__init__()
        
        # Image encoder (using ResNet50 as backbone)
        self.image_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()  # Remove final classification layer
        
        # Genomic feature encoder
        self.genomic_encoder = nn.Sequential(
            nn.Linear(len(BLOOD_CANCER_GENES), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head with proper initialization
        self.classification_head = nn.Sequential(
            nn.Linear(2048 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images, genomic_features):
        # Process images
        batch_size = images.size(0)
        image_features = self.image_encoder.conv1(images)
        image_features = self.image_encoder.bn1(image_features)
        image_features = self.image_encoder.relu(image_features)
        image_features = self.image_encoder.maxpool(image_features)
        
        image_features = self.image_encoder.layer1(image_features)
        image_features = self.image_encoder.layer2(image_features)
        image_features = self.image_encoder.layer3(image_features)
        image_features = self.image_encoder.layer4(image_features)
        
        # Generate segmentation
        segmentation = self.segmentation_head(image_features)
        
        # Process genomic features
        genomic_features = self.genomic_encoder(genomic_features)
        
        # Combine features for classification
        image_features = image_features.mean(dim=[2, 3])  # Global average pooling
        combined_features = torch.cat([image_features, genomic_features], dim=1)
        classification = self.classification_head(combined_features)
        
        return classification, segmentation

# Define data transforms
train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.0625, 0.0625),
        rotate=(-45, 45),
        p=0.5
    ),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, p=0.5),
    ], p=0.3),
    A.OneOf([
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Calculate train/val split
train_size = int(0.8 * BASE_SAMPLES)

def train_model_for_ratio(pos_ratio, neg_ratio, output_dir):
    """Train and evaluate model for a specific class balance ratio."""
    results = {}
    
    for strategy in IMBALANCE_STRATEGIES:
        print(f"\n🚀 Training model for ratio {pos_ratio}:{neg_ratio} with {strategy} strategy")
        
        # Generate dataset
        image_filenames, genomic_df = generate_dataset_for_ratio(
            pos_ratio, neg_ratio, BASE_SAMPLES, 
            os.path.join(output_dir, f"ratio_{pos_ratio}_{neg_ratio}")
        )
        
        # Create model directory
        model_dir = os.path.join(output_dir, f"model_ratio_{pos_ratio}_{neg_ratio}_{strategy}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Get imbalance handling components
        criterion, class_weights, sampler = get_imbalance_handler(strategy, pos_ratio, neg_ratio)
        
        # Dataset setup with strategy
        train_dataset = ImbalancedCancerDataset(
            image_filenames[:train_size],
            genomic_df.iloc[:train_size, :-1],
            genomic_df.iloc[:train_size]["Label"],
            transform=train_transform,
            strategy=strategy,
            is_train=True
        )
        
        val_dataset = ImbalancedCancerDataset(
            image_filenames[train_size:],
            genomic_df.iloc[train_size:, :-1],
            genomic_df.iloc[train_size:]["Label"],
            transform=val_transform,
            strategy='weighted',  # No sampling for validation
            is_train=False
        )
        
        # DataLoader setup
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Model training
        model = BloodCancerAnalyzer().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3
        )
        
        history = train_with_strategy(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device, model_dir, strategy
        )
        
        results[strategy] = history
    
    return results

def train_with_strategy(model, train_loader, val_loader, criterion, optimizer, 
                       scheduler, device, output_dir, strategy):
    """Train model with specific imbalance handling strategy."""
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'train_auc_pr': [], 'val_auc_pr': [], 'train_mcc': [], 'val_mcc': [],
        'train_specificity': [], 'val_specificity': [], 'train_g_mean': [], 'val_g_mean': [],
        'train_pos_acc': [], 'train_neg_acc': [], 'val_pos_acc': [], 'val_neg_acc': []
    }
    
    num_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, output_dir, strategy
        )
        
        # Validation
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, strategy
        )
        
        # Update history
        for k, v in train_metrics.items():
            if k in history:
                history[k].append(v)
        for k, v in val_metrics.items():
            if k in history:
                history[f'val_{k}'].append(v)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 
                      os.path.join(output_dir, f"best_model_{strategy}.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Generate visualizations
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            generate_epoch_visualizations(
                history, output_dir, epoch,
                train_loader, val_loader, model, device,
                strategy
            )
        
        # Print progress
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Train AUC-ROC: {train_metrics['auc_roc']:.4f}, Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    return history

def generate_comparative_analysis(all_results, base_output_dir):
    # Placeholder for comparative analysis code
    pass

def main():
    """Main function to run experiments with different class balances."""
    base_output_dir = "./experiments"
    os.makedirs(base_output_dir, exist_ok=True)
    
    all_results = {}
    
    # Train models for each ratio
    for pos_ratio, neg_ratio in BALANCE_RATIOS:
        print(f"\n{'='*50}")
        print(f"Starting experiment with ratio {pos_ratio}:{neg_ratio}")
        print(f"{'='*50}")
        
        history = train_model_for_ratio(
            pos_ratio, neg_ratio, base_output_dir
        )
        all_results[f"{pos_ratio}_{neg_ratio}"] = history
    
    # Generate comparative analysis
    generate_comparative_analysis(all_results, base_output_dir)

if __name__ == '__main__':
    main()
