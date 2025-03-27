import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
from face_networks import FaceEmbeddingNet, FaceEmbeddingNetWithL2, FaceSiameseNet
from pretrained_face_networks import FaceTripletNet
from face_losses import TripletLossWithMining, CenterLoss, SupConLoss, CircleLoss, TripletLoss


class FaceRecognitionTrainer:
    """
    Trainer class for face recognition models.
    
    This class handles training, validation, and testing of Siamese/Triplet networks
    for face recognition, supporting multiple loss functions and mining strategies.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[torch.device] = None,
                 model_dir: str = 'models',
                 mining_strategy: str = 'semi-hard',
                 loss_type: str = 'triplet',
                 embedding_dim: int = 128,
                 batch_size: int = 32):
        """
        Initialize the face recognition trainer.
        
        Args:
            model: Neural network model to train
            criterion: Loss function
            optimizer: Optimizer for parameter updates
            scheduler: Optional learning rate scheduler
            device: Device to train on (if None, will use CUDA if available)
            model_dir: Directory to save models to
            mining_strategy: Strategy for mining triplets ('all', 'hard', 'semi-hard')
            loss_type: Type of loss function ('triplet', 'center', 'supcon', 'circle')
            embedding_dim: Dimension of face embeddings
            batch_size: Batch size for training
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.mining_strategy = mining_strategy
        self.loss_type = loss_type
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Move model to device
        self.model.to(self.device)
        
        # Create model dir if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of metrics for this epoch
        """
        # Set model to training mode
        self.model.train()
        
        # Initialize metrics
        running_loss = 0.0
        num_correct = 0
        num_total = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        # Process each batch
        for batch_idx, batch in enumerate(pbar):
            # Handle different types of batches based on loss type
            if self.loss_type == 'triplet' or isinstance(self.model, FaceTripletNet):
                # Triplet loss with (anchor, positive, negative) triplets
                if len(batch) == 4:  # (anchor, positive, negative, labels)
                    anchor, positive, negative, labels = batch
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    if isinstance(self.model, FaceTripletNet):
                        # The pretrained_face_networks.py FaceTripletNet expects a tuple of tensors
                        embeddings = self.model((anchor, positive, negative))
                    else:
                        # Get embeddings for each image
                        anchor_embedding = self.model(anchor)
                        positive_embedding = self.model(positive)
                        negative_embedding = self.model(negative)
                        embeddings = (anchor_embedding, positive_embedding, negative_embedding)
                    
                    # Compute loss
                    loss = self.criterion(*embeddings)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    running_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
                    
                else:  # Online triplet loss with batch of embeddings and labels
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass to get embeddings
                    embeddings = self.model(data)
                    
                    # Compute loss and number of triplets
                    if hasattr(self.criterion, 'mining_strategy'):
                        loss, metrics = self.criterion(embeddings, labels)
                        num_triplets = metrics['num_triplets']
                    else:
                        loss = self.criterion(embeddings, labels)
                        num_triplets = 0
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    running_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item(), 'triplets': num_triplets})
            
            elif self.loss_type == 'pair' or isinstance(self.model, FaceSiameseNet):
                # Contrastive loss with (image1, image2, label) pairs
                if len(batch) == 3:  # (image1, image2, label)
                    img1, img2, labels = batch
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    if isinstance(self.model, FaceSiameseNet):
                        output1, output2 = self.model(img1, img2)
                    else:
                        output1 = self.model(img1)
                        output2 = self.model(img2)
                    
                    # Compute loss
                    loss = self.criterion(output1, output2, labels)
                    
                    # Compute accuracy for pairs
                    # Positive pairs: distance small, negative pairs: distance large
                    distances = torch.sum((output1 - output2).pow(2), dim=1)
                    predictions = (distances < 0.5).float()  # Threshold at 0.5
                    num_correct += torch.sum((predictions == labels).float()).item()
                    num_total += labels.size(0)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    running_loss += loss.item()
                    num_batches += 1
                    
                    # Calculate accuracy
                    accuracy = num_correct / max(1, num_total)
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item(), 'acc': accuracy})
                    
                else:  # Regular batch with center loss or other self-supervised losses
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    embeddings = self.model(data)
                    
                    # Compute loss
                    loss = self.criterion(embeddings, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Special handling for center loss which needs manual center update
                    if isinstance(self.criterion, CenterLoss):
                        self.criterion.update_centers(embeddings.detach(), labels)
                    
                    # Update metrics
                    running_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
            
            elif self.loss_type in ['center', 'supcon', 'circle']:
                # Self-supervised learning with a batch of embeddings and labels
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                embeddings = self.model(data)
                
                # Compute loss
                loss = self.criterion(embeddings, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Special handling for center loss which needs manual center update
                if isinstance(self.criterion, CenterLoss):
                    self.criterion.update_centers(embeddings.detach(), labels)
                
                # Update metrics
                running_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
        # Update learning rate scheduler if provided
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Calculate metrics
        avg_loss = running_loss / max(1, num_batches)
        accuracy = num_correct / max(1, num_total) if num_total > 0 else 0.0
        
        # Return metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
        }
        
        return metrics
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        running_loss = 0.0
        num_correct = 0
        num_total = 0
        num_batches = 0
        all_distances = []
        all_labels = []
        
        # Disable gradients for validation
        with torch.no_grad():
            # Progress bar
            pbar = tqdm(val_loader, desc='Validation', leave=False)
            
            # Process each batch
            for batch_idx, batch in enumerate(pbar):
                # Handle different types of batches based on loss type
                if self.loss_type == 'triplet' or isinstance(self.model, FaceTripletNet):
                    # Triplet loss with (anchor, positive, negative) triplets
                    if len(batch) == 4:  # (anchor, positive, negative, labels)
                        anchor, positive, negative, labels = batch
                        anchor = anchor.to(self.device)
                        positive = positive.to(self.device)
                        negative = negative.to(self.device)
                        
                        # Forward pass
                        if isinstance(self.model, FaceTripletNet):
                            # The pretrained_face_networks.py FaceTripletNet expects a tuple of tensors
                            embeddings = self.model((anchor, positive, negative))
                        else:
                            # Get embeddings for each image
                            anchor_embedding = self.model(anchor)
                            positive_embedding = self.model(positive)
                            negative_embedding = self.model(negative)
                            embeddings = (anchor_embedding, positive_embedding, negative_embedding)
                        
                        # Compute loss
                        loss = self.criterion(*embeddings)
                        
                        # Compute distances for evaluation metrics
                        anchor_embedding, positive_embedding, negative_embedding = embeddings
                        pos_distances = torch.sum((anchor_embedding - positive_embedding).pow(2), dim=1)
                        neg_distances = torch.sum((anchor_embedding - negative_embedding).pow(2), dim=1)
                        
                        # Create binary labels for distance pairs (1 for positive, 0 for negative)
                        distances = torch.cat([pos_distances, neg_distances])
                        pair_labels = torch.cat([torch.ones_like(pos_distances), torch.zeros_like(neg_distances)])
                        
                        # Store for precision-recall calculation
                        all_distances.append(distances.cpu().numpy())
                        all_labels.append(pair_labels.cpu().numpy())
                        
                        # Update metrics
                        running_loss += loss.item()
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({'loss': loss.item()})
                    
                    else:  # Online triplet loss with batch of embeddings and labels
                        data, labels = batch
                        data, labels = data.to(self.device), labels.to(self.device)
                        
                        # Forward pass to get embeddings
                        embeddings = self.model(data)
                        
                        # Compute loss
                        if hasattr(self.criterion, 'mining_strategy'):
                            loss, metrics = self.criterion(embeddings, labels)
                        else:
                            loss = self.criterion(embeddings, labels)
                        
                        # Update metrics
                        running_loss += loss.item()
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({'loss': loss.item()})
                
                elif self.loss_type == 'pair' or isinstance(self.model, FaceSiameseNet):
                    # Contrastive loss with (image1, image2, label) pairs
                    if len(batch) == 3:  # (image1, image2, label)
                        img1, img2, labels = batch
                        img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                        
                        # Forward pass
                        if isinstance(self.model, FaceSiameseNet):
                            output1, output2 = self.model(img1, img2)
                        else:
                            output1 = self.model(img1)
                            output2 = self.model(img2)
                        
                        # Compute loss
                        loss = self.criterion(output1, output2, labels)
                        
                        # Compute distances
                        distances = torch.sum((output1 - output2).pow(2), dim=1)
                        
                        # Store for precision-recall calculation
                        all_distances.append(distances.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                        
                        # Compute accuracy for pairs
                        # Positive pairs: distance small, negative pairs: distance large
                        predictions = (distances < 0.5).float()  # Threshold at 0.5
                        num_correct += torch.sum((predictions == labels).float()).item()
                        num_total += labels.size(0)
                        
                        # Update metrics
                        running_loss += loss.item()
                        num_batches += 1
                        
                        # Calculate accuracy
                        accuracy = num_correct / max(1, num_total)
                        
                        # Update progress bar
                        pbar.set_postfix({'loss': loss.item(), 'acc': accuracy})
                        
                    else:  # Regular batch with center loss or other self-supervised losses
                        data, labels = batch
                        data, labels = data.to(self.device), labels.to(self.device)
                        
                        # Forward pass
                        embeddings = self.model(data)
                        
                        # Compute loss
                        loss = self.criterion(embeddings, labels)
                        
                        # Update metrics
                        running_loss += loss.item()
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({'loss': loss.item()})
                
                elif self.loss_type in ['center', 'supcon', 'circle']:
                    # Self-supervised learning with a batch of embeddings and labels
                    data, labels = batch
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    embeddings = self.model(data)
                    
                    # Compute loss
                    loss = self.criterion(embeddings, labels)
                    
                    # Update metrics
                    running_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = running_loss / max(1, num_batches)
        accuracy = num_correct / max(1, num_total) if num_total > 0 else 0.0
        
        # Compute average precision if we have distance data
        avg_precision = 0.0
        if all_distances and all_labels:
            # Concatenate all distances and labels
            try:
                all_distances = np.concatenate(all_distances)
                all_labels = np.concatenate(all_labels)
                
                # Ensure we have enough data for meaningful metrics
                if len(all_distances) > 1 and len(np.unique(all_labels)) > 1:
                    # For PR curve, smaller distances should correspond to positive pairs
                    # So we negate the distances
                    precision, recall, _ = precision_recall_curve(all_labels, -all_distances)
                    avg_precision = average_precision_score(all_labels, -all_distances)
                else:
                    print("Not enough data for precision-recall metrics")
            except Exception as e:
                print(f"Error computing precision-recall metrics: {e}")
                # Create an empty metric for this batch
                precision, recall = np.array([]), np.array([])
                avg_precision = 0.0
        
        # Return metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'avg_precision': avg_precision,
        }
        
        return metrics
    
    def fit(self, 
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            num_epochs: int = 10,
            checkpoint_freq: int = 1) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            checkpoint_freq: Frequency of saving model checkpoints
            
        Returns:
            Training history
        """
        # Initialize best validation metrics
        best_val_loss = float('inf')
        best_val_precision = 0.0
        
        # Train for specified number of epochs
        for epoch in range(num_epochs):
            # Print epoch info
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}", end=' | ')
            if 'accuracy' in train_metrics and train_metrics['accuracy'] > 0:
                print(f"Train Acc: {train_metrics['accuracy']:.4f}", end=' | ')
            
            print(f"Val Loss: {val_metrics['loss']:.4f}", end=' | ')
            if 'accuracy' in val_metrics and val_metrics['accuracy'] > 0:
                print(f"Val Acc: {val_metrics['accuracy']:.4f}", end=' | ')
            
            if 'avg_precision' in val_metrics and val_metrics['avg_precision'] > 0:
                print(f"Val AP: {val_metrics['avg_precision']:.4f}", end='')
            
            print()  # New line
            
            # Update training history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            
            if 'accuracy' in train_metrics:
                self.history['train_acc'].append(train_metrics['accuracy'])
            if 'accuracy' in val_metrics:
                self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Save checkpoint if validation loss improved
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_loss')
                print(f"Saved model with best validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint if validation precision improved
            if 'avg_precision' in val_metrics and val_metrics['avg_precision'] > best_val_precision:
                best_val_precision = val_metrics['avg_precision']
                self.save_checkpoint('best_precision')
                print(f"Saved model with best validation precision: {best_val_precision:.4f}")
            
            # Save checkpoint at specified frequency
            if (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")
                print(f"Saved checkpoint for epoch {epoch+1}")
        
        return self.history
    
    def test(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Test the model on the test set.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary of test metrics
        """
        # Just use the validation logic for testing
        test_metrics = self.validate(test_loader)
        
        # Print test metrics
        print(f"Test Loss: {test_metrics['loss']:.4f}", end=' | ')
        if 'accuracy' in test_metrics and test_metrics['accuracy'] > 0:
            print(f"Test Acc: {test_metrics['accuracy']:.4f}", end=' | ')
        
        if 'avg_precision' in test_metrics and test_metrics['avg_precision'] > 0:
            print(f"Test AP: {test_metrics['avg_precision']:.4f}", end='')
        
        print()  # New line
        
        return test_metrics
    
    def save_checkpoint(self, checkpoint_name: str):
        """
        Save a model checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to save
        """
        checkpoint_path = os.path.join(self.model_dir, f"{checkpoint_name}.pth")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_type': self.loss_type,
            'mining_strategy': self.mining_strategy,
            'embedding_dim': self.embedding_dim,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint to load
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other parameters if they exist
        if 'loss_type' in checkpoint:
            self.loss_type = checkpoint['loss_type']
        if 'mining_strategy' in checkpoint:
            self.mining_strategy = checkpoint['mining_strategy']
        if 'embedding_dim' in checkpoint:
            self.embedding_dim = checkpoint['embedding_dim']
    
    def plot_history(self):
        """
        Plot training history.
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot accuracy if it exists
        if self.history['train_acc'] and self.history['val_acc']:
            ax2.plot(self.history['train_acc'], label='Train Acc')
            ax2.plot(self.history['val_acc'], label='Val Acc')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Save figure
        os.makedirs(self.model_dir, exist_ok=True)
        fig.savefig(os.path.join(self.model_dir, 'training_history.png'))
        

def create_trainer(model_type: str = 'triplet',
                  loss_type: str = 'triplet',
                  mining_strategy: str = 'semi-hard',
                  embedding_dim: int = 128,
                  lr: float = 0.001,
                  margin: float = 1.0,
                  num_classes: int = 100,
                  device: Optional[torch.device] = None) -> FaceRecognitionTrainer:
    """
    Create a trainer for face recognition.
    
    Args:
        model_type: Type of model ('triplet' or 'siamese')
        loss_type: Type of loss ('triplet', 'center', 'supcon', 'circle')
        mining_strategy: Strategy for mining triplets ('all', 'hard', 'semi-hard')
        embedding_dim: Dimension of the embedding vector
        lr: Learning rate
        margin: Margin for triplet/contrastive loss
        num_classes: Number of identity classes
        device: Device to train on
        
    Returns:
        A trainer object
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base embedding model
    if model_type in ['triplet', 'siamese']:
        embedding_net = FaceEmbeddingNetWithL2(embedding_dim=embedding_dim)
    else:
        embedding_net = FaceEmbeddingNet(embedding_dim=embedding_dim)
    
    # Create full model
    if model_type == 'triplet':
        model = FaceTripletNet(embedding_net)
    elif model_type == 'siamese':
        model = FaceSiameseNet(embedding_net)
    else:
        model = embedding_net
    
    # Create loss function
    if loss_type == 'triplet':
        criterion = TripletLossWithMining(margin=margin, mining_strategy=mining_strategy)
    elif loss_type == 'center':
        criterion = CenterLoss(num_classes=num_classes, feature_dim=embedding_dim)
    elif loss_type == 'supcon':
        criterion = SupConLoss(temperature=0.07)
    elif loss_type == 'circle':
        criterion = CircleLoss(margin=0.25, gamma=256)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Create optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Create learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Create trainer
    trainer = FaceRecognitionTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mining_strategy=mining_strategy,
        loss_type=loss_type,
        embedding_dim=embedding_dim
    )
    
    return trainer 