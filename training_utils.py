import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_utils import (
    plot_wbc_samples, plot_training_history,
    plot_genomic_features, plot_roc_curve,
    plot_confusion_matrix, visualize_wbc_types,
    save_visualization
)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive set of metrics."""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp)
    g_mean = np.sqrt(recall * specificity)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # Class-wise accuracy
    pos_acc = tp / (tp + fn)
    neg_acc = tn / (tn + fp)
    balanced_acc = (pos_acc + neg_acc) / 2
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'g_mean': g_mean,
        'mcc': mcc,
        'auc_pr': auc_pr,
        'auc_roc': auc_roc,
        'balanced_acc': balanced_acc,
        'pos_acc': pos_acc,
        'neg_acc': neg_acc,
        'confusion_matrix': {
            'tn': tn, 'fp': fp,
            'fn': fn, 'tp': tp
        }
    }

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, 
                output_dir, strategy):
    """Train for one epoch with comprehensive metrics."""
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    for batch_idx, (images, masks, genomic_features, labels) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        genomic_features = genomic_features.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        # Forward pass
        classifications, segmentations = model(images, genomic_features)
        
        # Calculate losses
        loss = criterion(classifications, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store predictions and labels
        probs = torch.sigmoid(classifications)
        preds = (probs > 0.5).float()
        
        total_loss += loss.item()
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Visualize first batch
        if batch_idx == 0:
            fig = plot_wbc_samples(images[:5], masks[:5], segmentations[:5])
            save_visualization(fig, os.path.join(output_dir, f'epoch_{epoch+1}_samples.png'))
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def validate_epoch(model, val_loader, criterion, device, strategy):
    """Validate with comprehensive metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks, genomic_features, labels in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            genomic_features = genomic_features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            classifications, segmentations = model(images, genomic_features)
            loss = criterion(classifications, labels)
            
            probs = torch.sigmoid(classifications)
            preds = (probs > 0.5).float()
            
            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def generate_epoch_visualizations(history, output_dir, epoch, train_loader, 
                                val_loader, model, device, strategy):
    """Generate comprehensive visualizations for the current epoch."""
    vis_dir = os.path.join(output_dir, f'epoch_{epoch+1}_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Training history plots
    metrics_to_plot = [
        ('loss', 'Loss'),
        ('balanced_acc', 'Balanced Accuracy'),
        ('f1', 'F1 Score'),
        ('auc_roc', 'ROC AUC'),
        ('auc_pr', 'PR AUC'),
        ('g_mean', 'G-Mean'),
        ('mcc', 'Matthews Correlation Coefficient')
    ]
    
    for metric, title in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label='Train')
        plt.plot(history[f'val_{metric}'], label='Validation')
        plt.title(f'{title} Over Time ({strategy})')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        save_visualization(plt.gcf(), os.path.join(vis_dir, f'{metric}_history.png'))
    
    # Class-wise accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_pos_acc'], label='Train Positive')
    plt.plot(history['train_neg_acc'], label='Train Negative')
    plt.plot(history['val_pos_acc'], label='Val Positive')
    plt.plot(history['val_neg_acc'], label='Val Negative')
    plt.title(f'Class-wise Accuracy Over Time ({strategy})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    save_visualization(plt.gcf(), os.path.join(vis_dir, 'class_wise_acc.png'))
    
    # Get validation predictions
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks, genomic_features, labels in val_loader:
            images = images.to(device)
            genomic_features = genomic_features.to(device)
            classifications, _ = model(images, genomic_features)
            probs = torch.sigmoid(classifications)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion matrix heatmap
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({strategy})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_visualization(plt.gcf(), os.path.join(vis_dir, 'confusion_matrix.png'))
    
    # ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    ax1.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.3f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve ({strategy})')
    ax1.legend()
    ax1.grid(True)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    ax2.plot(recall, precision, label=f'AP = {average_precision_score(all_labels, all_probs):.3f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve ({strategy})')
    ax2.legend()
    ax2.grid(True)
    
    save_visualization(fig, os.path.join(vis_dir, 'roc_pr_curves.png'))

def generate_comparative_analysis(all_results, base_output_dir):
    """Generate enhanced comparative analysis across different strategies and ratios."""
    analysis_dir = os.path.join(base_output_dir, 'comparative_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    metrics_to_compare = [
        'balanced_acc', 'f1', 'auc_roc', 'auc_pr',
        'g_mean', 'mcc', 'precision', 'recall'
    ]
    
    # Strategy comparison for each ratio
    for ratio in all_results:
        ratio_dir = os.path.join(analysis_dir, f'ratio_{ratio}')
        os.makedirs(ratio_dir, exist_ok=True)
        
        for metric in metrics_to_compare:
            plt.figure(figsize=(12, 6))
            for strategy, history in all_results[ratio].items():
                plt.plot(history[f'val_{metric}'], label=strategy)
            plt.title(f'{metric} Comparison Across Strategies (Ratio {ratio})')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            save_visualization(plt.gcf(), os.path.join(ratio_dir, f'{metric}_comparison.png'))
    
    # Final performance heatmap
    strategies = list(next(iter(all_results.values())).keys())
    ratios = list(all_results.keys())
    
    for metric in metrics_to_compare:
        performance_matrix = np.zeros((len(ratios), len(strategies)))
        
        for i, ratio in enumerate(ratios):
            for j, strategy in enumerate(strategies):
                performance_matrix[i, j] = max(all_results[ratio][strategy][f'val_{metric}'])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(performance_matrix, annot=True, fmt='.3f',
                    xticklabels=strategies, yticklabels=ratios,
                    cmap='YlOrRd')
        plt.title(f'Best {metric} Performance')
        plt.xlabel('Strategy')
        plt.ylabel('Class Ratio')
        save_visualization(plt.gcf(), os.path.join(analysis_dir, f'{metric}_heatmap.png'))
    
    # Generate summary report
    with open(os.path.join(analysis_dir, 'summary_report.txt'), 'w') as f:
        f.write("Performance Summary\n")
        f.write("="*50 + "\n\n")
        
        for ratio in ratios:
            f.write(f"\nClass Ratio: {ratio}\n")
            f.write("-"*30 + "\n")
            
            for strategy in strategies:
                f.write(f"\nStrategy: {strategy}\n")
                history = all_results[ratio][strategy]
                
                for metric in metrics_to_compare:
                    best_value = max(history[f'val_{metric}'])
                    final_value = history[f'val_{metric}'][-1]
                    f.write(f"{metric}:\n")
                    f.write(f"  Best: {best_value:.4f}\n")
                    f.write(f"  Final: {final_value:.4f}\n") 