import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

def plot_wbc_samples(images, masks, predictions=None, num_samples=5):
    """Plot WBC images with their segmentation masks and predictions."""
    fig, axes = plt.subplots(num_samples, 3 if predictions is not None else 2, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        if predictions is not None:
            # Predicted mask
            axes[i, 2].imshow(torch.sigmoid(predictions[i]).squeeze().cpu().detach().numpy(), cmap='gray')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_genomic_features(genomic_data, labels):
    """Plot genomic feature distributions."""
    plt.figure(figsize=(15, 10))
    
    # Convert to DataFrame if not already
    if isinstance(genomic_data, torch.Tensor):
        genomic_data = genomic_data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    df = pd.DataFrame(genomic_data)
    df['Label'] = labels
    
    # Create violin plots for each gene
    sns.violinplot(data=df.melt(id_vars=['Label']), 
                  x='variable', y='value', hue='Label',
                  split=True)
    plt.xticks(rotation=45)
    plt.title('Gene Expression Distribution by Class')
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_pred):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), y_pred.cpu().numpy())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true.cpu().numpy(), (y_pred > 0.5).cpu().numpy())
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

def visualize_wbc_types(image, segmentation, wbc_types):
    """Visualize different types of WBCs in the image."""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmentation mask
    plt.subplot(132)
    plt.imshow(segmentation.squeeze().cpu().numpy(), cmap='nipy_spectral')
    plt.title('WBC Segmentation')
    plt.axis('off')
    
    # WBC type distribution
    plt.subplot(133)
    plt.pie([v for v in wbc_types.values()], 
            labels=[k for k in wbc_types.keys()],
            autopct='%1.1f%%')
    plt.title('WBC Type Distribution')
    
    plt.tight_layout()
    return plt.gcf()

def save_visualization(fig, filename):
    """Save visualization figure to file."""
    fig.savefig(filename)
    plt.close(fig) 