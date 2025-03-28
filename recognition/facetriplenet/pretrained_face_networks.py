#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pretrained Face Network Models

This module provides network architectures that leverage pretrained models
for faster training and better performance on face recognition tasks.
The models can be fine-tuned on face datasets with various loss functions.

Author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, MobileNet_V2_Weights
import timm
from typing import Tuple


class PretrainedFaceEmbedding(nn.Module):
    """
    Face embedding network using a pretrained backbone.
    
    This network takes a face image and outputs an embedding vector.
    It uses a pretrained backbone (e.g., ResNet, EfficientNet) as feature extractor
    and adds a projection head to get the final embedding.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 128, 
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize the face embedding network.
        
        Args:
            embedding_dim (int): Dimension of the output embedding
            model_name (str): Name of the pretrained model to use
                Options: 'resnet18', 'resnet50', 'resnet101', 'efficientnet_b0', 'mobilenetv2', 'vit_small'
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze the backbone weights
        """
        super(PretrainedFaceEmbedding, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Get backbone and input feature dimension
        self.backbone, self.in_features = self._get_backbone(model_name, pretrained)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.in_features, self.in_features // 2),
            nn.ReLU(),
            nn.Linear(self.in_features // 2, embedding_dim)
        )
        
        # Freeze backbone weights if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _get_backbone(self, model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Get the backbone model and its output feature dimension.
        
        Args:
            model_name (str): Name of the pretrained model
            pretrained (bool): Whether to use pretrained weights
            
        Returns:
            tuple: (backbone model, output feature dimension)
        """
        # ResNet models
        if model_name == 'resnet18':
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()  # Remove the final FC layer
            return model, in_features
            
        elif model_name == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()  # Remove the final FC layer
            return model, in_features
            
        elif model_name == 'resnet101':
            model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.fc.in_features
            model.fc = nn.Identity()  # Remove the final FC layer
            return model, in_features
            
        # EfficientNet models
        elif model_name.startswith('efficientnet'):
            # Using timm for efficient-net models
            model = timm.create_model(model_name, pretrained=pretrained)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()  # Remove the classifier
            return model, in_features
            
        # MobileNetV2
        elif model_name == 'mobilenetv2':
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()  # Remove the classifier
            return model, in_features
            
        # Vision Transformer (ViT)
        elif model_name.startswith('vit'):
            model = timm.create_model(model_name, pretrained=pretrained)
            in_features = model.head.in_features
            model.head = nn.Identity()  # Remove the head
            return model, in_features
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Forward pass through the backbone
        features = self.backbone(x)
        
        # Forward pass through the projection head
        embedding = self.projection(features)
        
        return embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding for an input tensor.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embedding tensor
        """
        return self.forward(x)


class PretrainedFaceEmbeddingWithL2(PretrainedFaceEmbedding):
    """
    Face embedding network with L2 normalization on the output embeddings.
    
    This network extends PretrainedFaceEmbedding to normalize the output embeddings
    to have unit L2 norm, which is useful for metric learning.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: L2-normalized embedding tensor
        """
        # Get embeddings from parent class
        embedding = super(PretrainedFaceEmbeddingWithL2, self).forward(x)
        
        # Apply L2 normalization
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class FaceTripletNet(nn.Module):
    """
    Network for processing triplets of face images.
    
    This network takes triplets of face images (anchor, positive, negative)
    and outputs their embeddings using a shared face embedding network.
    """
    
    def __init__(self, embedding_net: nn.Module):
        """
        Initialize the triplet network.
        
        Args:
            embedding_net (nn.Module): The embedding network to use
        """
        super(FaceTripletNet, self).__init__()
        self.embedding_net = embedding_net
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x (tuple): Tuple of (anchor, positive, negative) tensors
            
        Returns:
            tuple: Tuple of (anchor_embedding, positive_embedding, negative_embedding)
        """
        anchor, positive, negative = x
        
        # Get embeddings for each image
        anchor_embedding = self.embedding_net(anchor)
        positive_embedding = self.embedding_net(positive)
        negative_embedding = self.embedding_net(negative)
        
        return anchor_embedding, positive_embedding, negative_embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for a single input.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embedding tensor
        """
        return self.embedding_net(x) 