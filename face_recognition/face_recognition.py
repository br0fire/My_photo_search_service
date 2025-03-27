#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Face Recognition System

This script provides both the model definitions and evaluation functionality for face recognition
in a single file. It supports pretrained backbones (ResNet, EfficientNet, etc.) and includes
functions for evaluating models on binary classification tasks.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, MobileNet_V2_Weights
import timm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import Tuple


# ======== Model Architecture ========

class PretrainedFaceEmbedding(nn.Module):
    """Face embedding network using a pretrained backbone"""
    
    def __init__(
        self, 
        embedding_dim: int = 128, 
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(PretrainedFaceEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        self.backbone, self.in_features = self._get_backbone(model_name, pretrained)
        
        self.projection = nn.Sequential(
            nn.Linear(self.in_features, self.in_features // 2),
            nn.ReLU(),
            nn.Linear(self.in_features // 2, embedding_dim)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _get_backbone(self, model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        if model_name == 'resnet18':
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_features
            
        elif model_name == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_features
            
        elif model_name == 'resnet101':
            model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_features
            
        elif model_name.startswith('efficientnet'):
            model = timm.create_model(model_name, pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()
            return model, in_features
            
        elif model_name == 'mobilenetv2':
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, in_features
            
        elif model_name.startswith('vit'):
            model = timm.create_model(model_name, pretrained=pretrained)
            in_features = model.head.in_features
            model.head = nn.Identity()
            return model, in_features
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.projection(features)
        return embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class PretrainedFaceEmbeddingWithL2(PretrainedFaceEmbedding):
    """Face embedding network with L2 normalization"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = super(PretrainedFaceEmbeddingWithL2, self).forward(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class FaceTripletNet(nn.Module):
    """Network for processing triplets of face images"""
    
    def __init__(self, embedding_net: nn.Module):
        super(FaceTripletNet, self).__init__()
        self.embedding_net = embedding_net
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, positive, negative = x
        
        anchor_embedding = self.embedding_net(anchor)
        positive_embedding = self.embedding_net(positive)
        negative_embedding = self.embedding_net(negative)
        
        return anchor_embedding, positive_embedding, negative_embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_net(x)


# ======== Evaluation Functionality ========

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating) or isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def load_model(model_path, device):
    """Load a trained face recognition model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint.get('model_name', 'resnet50')
    embedding_dim = checkpoint.get('embedding_dim', 128)
    freeze_backbone = checkpoint.get('freeze_backbone', False)
    
    embedding_net = PretrainedFaceEmbeddingWithL2(
        embedding_dim=embedding_dim,
        model_name=model_name,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    model = FaceTripletNet(embedding_net)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def load_and_preprocess_images(folder_path, transform):
    """Load all images from a folder and preprocess them"""
    images = []
    folder = Path(folder_path)
    
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    for img_file in image_files:
        img_path = folder / img_file
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            images.append((tensor, img_file))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return images


def compute_embeddings(model, images, device):
    """Compute embeddings for a list of images"""
    embeddings = {}
    
    with torch.no_grad():
        for tensor, filename in tqdm(images, desc="Computing embeddings"):
            tensor = tensor.to(device)
            embedding = model.get_embedding(tensor)
            embeddings[filename] = embedding.cpu().numpy().flatten()
    
    return embeddings


def compute_distances_and_labels(class0_embeddings, class1_embeddings):
    """Compute distances between all pairs of embeddings and their corresponding labels"""
    distances = []
    labels = []
    
    class0_keys = list(class0_embeddings.keys())
    for i in range(len(class0_keys)):
        for j in range(i+1, len(class0_keys)):
            emb1 = class0_embeddings[class0_keys[i]]
            emb2 = class0_embeddings[class0_keys[j]]
            distance = np.sum((emb1 - emb2) ** 2)
            distances.append(distance)
            labels.append(0)  # Different individuals (negative pair)
    
    class1_keys = list(class1_embeddings.keys())
    for i in range(len(class1_keys)):
        for j in range(i+1, len(class1_keys)):
            emb1 = class1_embeddings[class1_keys[i]]
            emb2 = class1_embeddings[class1_keys[j]]
            distance = np.sum((emb1 - emb2) ** 2)
            distances.append(distance)
            labels.append(1)  # Same individual (positive pair)
    
    return np.array(distances), np.array(labels)


def find_best_threshold(distances, labels):
    """Find the best threshold for classification based on F1 score"""
    precision, recall, thresholds_pr = precision_recall_curve(labels, -distances)
    
    f1_scores = []
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            f1_scores.append((f1, i))
    
    best_f1, best_idx = max(f1_scores, key=lambda x: x[0])
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    if best_idx < len(thresholds_pr):
        best_threshold = -thresholds_pr[best_idx]
    else:
        best_threshold = -thresholds_pr[-1]
    
    predictions = (distances < best_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    metrics = {
        'threshold': best_threshold,
        'accuracy': accuracy,
        'precision': best_precision,
        'recall': best_recall,
        'f1_score': best_f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    return best_threshold, metrics


def plot_metrics(distances, labels, best_threshold, output_path):
    """Plot ROC curve, precision-recall curve, and distance distributions"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    fpr, tpr, _ = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    
    precision, recall, _ = precision_recall_curve(labels, -distances)
    ax2.plot(recall, precision, color='blue', lw=2)
    ax2.axvline(x=recall[np.argmin(np.abs(-distances - best_threshold))], color='red', linestyle='--', label=f'Best threshold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    class0_distances = distances[labels == 0]
    class1_distances = distances[labels == 1]
    
    ax3.hist(class0_distances, alpha=0.5, bins=50, label='Class 0 (Different)')
    ax3.hist(class1_distances, alpha=0.5, bins=50, label='Class 1 (Same)')
    ax3.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best threshold: {best_threshold:.4f}')
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Count')
    ax3.set_title('Distance Distributions')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate(model_path, test_dir, output_dir='evaluation_results', threshold=None):
    """Evaluate a face recognition model on a binary classification task"""
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_dir = Path(test_dir)
    class0_dir = test_dir / '0'
    class1_dir = test_dir / '1'
    
    if not class0_dir.exists() or not class1_dir.exists():
        print(f"Error: Test directory must contain '0' and '1' subdirectories.")
        return
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Loading images from class 0 directory: {class0_dir}")
    class0_images = load_and_preprocess_images(class0_dir, transform)
    print(f"Loaded {len(class0_images)} images from class 0")
    
    print(f"Loading images from class 1 directory: {class1_dir}")
    class1_images = load_and_preprocess_images(class1_dir, transform)
    print(f"Loaded {len(class1_images)} images from class 1")
    
    print(f"Computing embeddings for all images...")
    class0_embeddings = compute_embeddings(model, class0_images, device)
    class1_embeddings = compute_embeddings(model, class1_images, device)
    
    print(f"Computing distances between embedding pairs...")
    distances, labels = compute_distances_and_labels(class0_embeddings, class1_embeddings)
    
    print(f"Computed {len(distances)} pairs for evaluation:")
    print(f"  - Negative pairs (different faces): {sum(labels == 0)}")
    print(f"  - Positive pairs (same face): {sum(labels == 1)}")
    
    if threshold is not None:
        print(f"Using custom threshold: {threshold}")
        best_threshold = threshold
        
        predictions = (distances < best_threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        metrics = {
            'threshold': best_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    else:
        print(f"Finding optimal threshold based on F1 score...")
        best_threshold, metrics = find_best_threshold(distances, labels)
    
    print("\n===== Evaluation Results =====")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(f"TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
    print(f"FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
    
    output_path = os.path.join(output_dir, 'evaluation_plots.png')
    plot_metrics(distances, labels, best_threshold, output_path)
    print(f"Evaluation plots saved to {output_path}")
    
    output_json = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
    print(f"Evaluation metrics saved to {output_json}")
    
    return metrics


def main():
    """Main function for the face recognition evaluation script"""
    parser = argparse.ArgumentParser(description='Face Recognition Binary Classification Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test-dir', type=str, required=True, help='Path to the test directory with 0 and 1 subfolders')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--threshold', type=float, help='Custom threshold value for classification (if not provided, optimal threshold will be determined automatically)')
    args = parser.parse_args()
    
    evaluate(args.model, args.test_dir, args.output_dir, args.threshold)


if __name__ == '__main__':
    main() 