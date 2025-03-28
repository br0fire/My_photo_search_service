import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class FaceEmbeddingNet(nn.Module):
    """
    A CNN-based network for extracting face embeddings.
    
    This network takes face images and projects them into an embedding space
    where faces of the same person are close together and faces of different
    people are far apart.
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = False):
        """
        Initialize the face embedding network.
        
        Args:
            embedding_dim: Dimension of the output embedding vector
            pretrained: Whether to use pretrained weights (if available)
        """
        super(FaceEmbeddingNet, self).__init__()
        
        # Feature extraction layers with larger receptive field for faces
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Adaptive pooling to handle different input image sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for embedding
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedded representation of the input
        """
        output = self.convnet(x)
        output = self.adaptive_pool(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding vector for the input.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedding vector
        """
        return self.forward(x)


class FaceEmbeddingNetWithL2(FaceEmbeddingNet):
    """
    Face embedding network that normalizes the output embedding to the unit hypersphere.
    This is often beneficial for face recognition tasks as we typically
    care about the direction of the embedding vector, not its magnitude.
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = False):
        """
        Initialize the L2-normalized face embedding network.
        
        Args:
            embedding_dim: Dimension of the output embedding vector
            pretrained: Whether to use pretrained weights (if available)
        """
        super(FaceEmbeddingNetWithL2, self).__init__(embedding_dim, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with L2 normalization.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            L2-normalized embedding vector
        """
        output = super(FaceEmbeddingNetWithL2, self).forward(x)
        output = F.normalize(output, p=2, dim=1)
        return output


class FaceSiameseNet(nn.Module):
    """
    Siamese network that processes a pair of face images through the same
    embedding network and outputs their embeddings.
    """
    
    def __init__(self, embedding_net: nn.Module):
        """
        Initialize the Siamese network.
        
        Args:
            embedding_net: The network used to extract embeddings
        """
        super(FaceSiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a pair of inputs through the Siamese network.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Tuple of (embedding1, embedding2)
        """
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding for a single input.
        
        Args:
            x: Input tensor
            
        Returns:
            Embedding vector
        """
        return self.embedding_net(x)


class FaceTripletNet(nn.Module):
    """
    Triplet network that processes triplets of face images (anchor, positive, negative)
    through the same embedding network and outputs their embeddings.
    """
    
    def __init__(self, embedding_net: nn.Module):
        """
        Initialize the triplet network.
        
        Args:
            embedding_net: The network used to extract embeddings
        """
        super(FaceTripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a triplet of inputs through the network.
        
        Args:
            x1: Anchor input tensor
            x2: Positive input tensor (same identity as anchor)
            x3: Negative input tensor (different identity from anchor)
            
        Returns:
            Tuple of (anchor_embedding, positive_embedding, negative_embedding)
        """
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the embedding for a single input.
        
        Args:
            x: Input tensor
            
        Returns:
            Embedding vector
        """
        return self.embedding_net(x)


class ArcFaceLayer(nn.Module):
    """
    ArcFace layer for improved face recognition accuracy.
    
    ArcFace adds an angular margin penalty between the feature vector and the
    weight vector of the correct class to enhance discriminative power.
    """
    
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        """
        Initialize the ArcFace layer.
        
        Args:
            in_features: Size of input features
            out_features: Number of classes
            s: Scale factor for logits
            m: Margin parameter for angular penalty
        """
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # For numerical stability
        self.eps = 1e-7
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(math.pi - m))
        self.mm = torch.sin(torch.tensor(math.pi - m)) * m

    def forward(self, input: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for ArcFace.
        
        Args:
            input: Feature vectors, shape (B, in_features)
            label: Ground truth labels, shape (B,)
            
        Returns:
            Logits with angular margin penalty if labels are provided, else cosine similarity
        """
        # Normalize features and weights
        x = F.normalize(input, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(x, w)
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)
        
        # If no labels provided, just return scaled cosine similarities
        if label is None:
            return self.s * cos_theta
        
        # Convert to one-hot encoding
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # Compute sin and cos
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        
        # Add angular margin
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # For numerical stability
        cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        
        # Apply margin only for the target class
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        output *= self.s
        
        return output 