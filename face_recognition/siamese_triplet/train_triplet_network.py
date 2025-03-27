#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Siamese Network with Triplet Loss

This script trains a Siamese network using triplet loss for face recognition.
It supports various pretrained backbone networks and different mining strategies.

Usage:
    python train_triplet_network.py --data-dir data --model resnet50 --mining-strategy semi-hard

Author: User
"""

import os
import sys
import argparse
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

from face_networks import FaceEmbeddingNetWithL2
from pretrained_face_networks import PretrainedFaceEmbeddingWithL2, FaceTripletNet
from face_losses import TripletLossWithMining, TripletLoss
from face_datasets import FaceImageDataset, FaceTripletDataset
from face_trainer import FaceRecognitionTrainer, create_trainer


def setup_logging(log_dir, name):
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def visualize_embeddings(model, dataset, n_samples=300, output_path=None):
    """Visualize embeddings using t-SNE."""
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Get a subset of samples for visualization
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    # Get embeddings
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for idx in indices:
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)
            embedding = model.get_embedding(img).cpu().numpy()
            embeddings.append(embedding[0])
            labels.append(label)
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    
    # Use different colors for different identities
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f"ID {label}", alpha=0.7)
    
    plt.title("t-SNE Visualization of Face Embeddings")
    plt.legend(loc='best', bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def compute_verification_metrics(model, dataset, n_pairs=1000, output_dir=None):
    """Compute verification metrics (precision-recall, accuracy)."""
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Create positive and negative pairs
    pos_pairs = []
    neg_pairs = []
    
    # Group images by identity
    identity_images = {}
    for i in range(len(dataset.identities)):
        identity_images[i] = dataset.get_identity_images(i)
    
    # Generate positive pairs (same identity)
    valid_identities = [idx for idx, images in identity_images.items() if len(images) >= 2]
    if not valid_identities:
        raise ValueError("No valid identities for verification")
    
    for _ in range(n_pairs // 2):
        # Select random identity with at least 2 images
        identity = np.random.choice(valid_identities)
        # Select two different images
        if len(identity_images[identity]) >= 2:
            idx1, idx2 = np.random.choice(identity_images[identity], 2, replace=False)
            pos_pairs.append((idx1, idx2))
    
    # Generate negative pairs (different identities)
    for _ in range(n_pairs // 2):
        # Select two different identities
        if len(valid_identities) >= 2:
            identity1, identity2 = np.random.choice(valid_identities, 2, replace=False)
            # Select one image from each identity
            idx1 = np.random.choice(identity_images[identity1])
            idx2 = np.random.choice(identity_images[identity2])
            neg_pairs.append((idx1, idx2))
    
    # Compute distances and labels for all pairs
    distances = []
    labels = []
    
    with torch.no_grad():
        # Process positive pairs
        for idx1, idx2 in pos_pairs:
            img1, _ = dataset[idx1]
            img2, _ = dataset[idx2]
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
            
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            
            # Compute Euclidean distance
            dist = torch.sum((emb1 - emb2) ** 2).item()
            distances.append(dist)
            labels.append(1)  # 1 for positive pair
        
        # Process negative pairs
        for idx1, idx2 in neg_pairs:
            img1, _ = dataset[idx1]
            img2, _ = dataset[idx2]
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
            
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            
            # Compute Euclidean distance
            dist = torch.sum((emb1 - emb2) ** 2).item()
            distances.append(dist)
            labels.append(0)  # 0 for negative pair
    
    # Convert to numpy arrays
    distances = np.array(distances)
    labels = np.array(labels)
    
    # For PR curve, smaller distances should correspond to positive pairs
    precision, recall, thresholds = precision_recall_curve(labels, -distances)
    ap = average_precision_score(labels, -distances)
    
    # Compute best accuracy at optimal threshold
    accuracies = []
    for threshold in thresholds:
        predictions = (-distances >= threshold).astype(int)
        accuracy = (predictions == labels).mean()
        accuracies.append(accuracy)
    
    best_accuracy = max(accuracies) if accuracies else 0
    best_threshold = thresholds[np.argmax(accuracies)] if accuracies else 0
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP={ap:.4f}, Acc={best_accuracy:.4f})')
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()
    else:
        plt.show()
    
    # Save metrics to JSON if output_dir is provided
    metrics = {
        'average_precision': float(ap),
        'best_accuracy': float(best_accuracy),
        'best_threshold': float(best_threshold)
    }
    
    if output_dir:
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Siamese Network with Triplet Loss')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to the dataset directory')
    parser.add_argument('--target-size', type=int, default=224,
                        help='Target size of the input images')
    parser.add_argument('--num-triplets', type=int, default=10000,
                        help='Number of triplets for training')
                        
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['custom', 'resnet18', 'resnet50', 'resnet101', 
                                'efficientnet_b0', 'mobilenetv2', 'vit_small'],
                        help='Model architecture to use')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Dimension of the embedding vector')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone weights')
                        
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.3,
                        help='Margin for triplet loss')
    parser.add_argument('--mining-strategy', type=str, default='semi-hard',
                        choices=['all', 'hard', 'semi-hard'],
                        help='Mining strategy for triplet loss')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--checkpoint-freq', type=int, default=5,
                        help='Frequency of saving checkpoints (epochs)')
                        
    # Misc arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Directory to save experiment results')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: model_mining-strategy_timestamp)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function for training the Siamese network."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device)
    
    # Create experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.model}_{args.mining_strategy}_{timestamp}"
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.exp_name)
    
    # Log experiment configuration
    logger.info("=" * 50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 50)
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("=" * 50)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info("Loading datasets...")
    
    # Data augmentation and transforms
    import torchvision.transforms as transforms
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((args.target_size, args.target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.target_size, args.target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FaceImageDataset(
        args.data_dir, 'train', train_transform, 
        target_size=(args.target_size, args.target_size)
    )
    val_dataset = FaceImageDataset(
        args.data_dir, 'valid', val_transform, 
        target_size=(args.target_size, args.target_size)
    )
    
    # Check if test dataset exists, if not split validation dataset
    test_dir = os.path.join(args.data_dir, 'test', 'images')
    if not os.path.exists(test_dir):
        logger.info("Test dataset not found. Splitting validation dataset into validation and test sets...")
        
        # Split validation dataset: 70% val and 30% test
        val_indices, test_indices = train_test_split(
            list(range(len(val_dataset))), 
            test_size=0.3, 
            stratify=[val_dataset.image_identities[i] for i in range(len(val_dataset))],
            random_state=42
        )
        
        # Create SubsetDataset
        from torch.utils.data import Subset
        test_dataset = Subset(val_dataset, test_indices)
        
        # Update validation dataset to be a subset
        val_dataset = Subset(val_dataset, val_indices)
        
        logger.info(f"Split validation dataset: {len(val_indices)} images for validation, {len(test_indices)} images for testing")
    else:
        # Load test dataset normally
        test_dataset = FaceImageDataset(
            args.data_dir, 'test', val_transform, 
            target_size=(args.target_size, args.target_size)
        )
    
    logger.info(f"Training dataset: {len(train_dataset)} images, {len(train_dataset.identities)} identities")
    logger.info(f"Validation dataset: {len(val_dataset)} images")
    logger.info(f"Test dataset: {len(test_dataset)} images")
    
    # Create triplet datasets
    train_triplet_dataset = FaceTripletDataset(
        train_dataset, num_triplets=args.num_triplets, triplet_type=args.mining_strategy
    )
    
    # Handle if validation dataset is a Subset
    if isinstance(val_dataset, torch.utils.data.Subset):
        from copy import deepcopy
        # Create a temporary dataset that works with FaceTripletDataset
        temp_val_dataset = deepcopy(val_dataset.dataset)
        # Filter image files and identities to include only validation images
        temp_val_dataset.image_files = [temp_val_dataset.image_files[i] for i in val_dataset.indices]
        temp_val_dataset.image_identities = [temp_val_dataset.image_identities[i] for i in val_dataset.indices]
        val_triplet_dataset = FaceTripletDataset(
            temp_val_dataset, num_triplets=args.num_triplets // 2, triplet_type=args.mining_strategy
        )
    else:
        val_triplet_dataset = FaceTripletDataset(
            val_dataset, num_triplets=args.num_triplets // 2, triplet_type=args.mining_strategy
        )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_triplet_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_triplet_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.info(f"Created data loaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    
    if args.model == 'custom':
        embedding_net = FaceEmbeddingNetWithL2(embedding_dim=args.embedding_dim)
    else:
        embedding_net = PretrainedFaceEmbeddingWithL2(
            embedding_dim=args.embedding_dim,
            model_name=args.model,
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
    
    model = FaceTripletNet(embedding_net)
    model = model.to(device)
    
    # Create loss function
    criterion = TripletLoss(margin=args.margin)
    
    # Create optimizer
    # If backbone is frozen, use higher learning rate
    lr = args.lr * 5 if args.freeze_backbone else args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr/10)
    else:
        scheduler = None
    
    # Create trainer
    trainer = FaceRecognitionTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_dir=output_dir,
        mining_strategy=args.mining_strategy,
        loss_type='triplet',
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size
    )
    
    # Train the model
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader, val_loader, 
        num_epochs=args.epochs, 
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(val) for val in values]
        json.dump(serializable_history, f, indent=4)
    
    # Plot training history
    trainer.plot_history()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Evaluate on test set
    logger.info("Evaluating on test dataset...")
    
    # Handle if test dataset is a Subset
    if isinstance(test_dataset, torch.utils.data.Subset):
        from copy import deepcopy
        # Create a temporary dataset that works with the evaluation functions
        temp_test_dataset = deepcopy(test_dataset.dataset)
        # Filter image files and identities to include only test images
        temp_test_dataset.image_files = [temp_test_dataset.image_files[i] for i in test_dataset.indices]
        temp_test_dataset.image_identities = [temp_test_dataset.image_identities[i] for i in test_dataset.indices]
        metrics = compute_verification_metrics(
            model, temp_test_dataset, output_dir=output_dir
        )
        
        # Also visualize embeddings with the temp dataset
        visualize_embeddings(
            model, temp_test_dataset, n_samples=min(300, len(temp_test_dataset)),
            output_path=os.path.join(output_dir, 'embeddings_tsne.png')
        )
    else:
        metrics = compute_verification_metrics(
            model, test_dataset, output_dir=output_dir
        )
        
        # Visualize embeddings
        logger.info("Visualizing embeddings...")
        visualize_embeddings(
            model, test_dataset, n_samples=min(300, len(test_dataset)),
            output_path=os.path.join(output_dir, 'embeddings_tsne.png')
        )
    
    logger.info(f"Test metrics: {metrics}")
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")


if __name__ == '__main__':
    main() 