import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class TripletLossWithMining(nn.Module):
    """
    Triplet loss with online mining strategy.
    
    This implements triplet loss with three different mining strategies:
    - Hard triplet mining: select the hardest negative samples
    - Semi-hard triplet mining: select negatives within the margin
    - All triplets: use all possible triplets
    """
    
    def __init__(self, margin: float = 1.0, mining_strategy: str = 'semi-hard'):
        """
        Initialize the triplet loss with a mining strategy.
        
        Args:
            margin: Margin to enforce between positive and negative distances
            mining_strategy: One of ['all', 'hard', 'semi-hard']
        """
        super(TripletLossWithMining, self).__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        if mining_strategy not in ['all', 'hard', 'semi-hard']:
            raise ValueError(f"Mining strategy '{mining_strategy}' not supported. Use one of ['all', 'hard', 'semi-hard']")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute triplet loss with the selected mining strategy.
        
        Args:
            embeddings: Feature embeddings of shape (batch_size, embedding_dim)
            labels: Identity labels of shape (batch_size)
            
        Returns:
            Tuple containing:
                - loss: Scalar loss value
                - metrics: Dictionary with additional information (num_triplets, etc.)
        """
        # Get distance matrix (squared Euclidean distances)
        dist_mat = self._get_distance_matrix(embeddings)
        
        # Get triplets based on mining strategy
        if self.mining_strategy == 'all':
            triplets = self._get_all_triplets(labels)
        else:
            triplets = self._get_mined_triplets(dist_mat, labels)
        
        if len(triplets) == 0:
            # Return zero loss if no triplets found
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True), {"num_triplets": 0}
        
        # Compute loss for selected triplets
        anchors, positives, negatives = triplets
        loss = self._compute_triplet_loss(dist_mat, anchors, positives, negatives)
        
        metrics = {"num_triplets": len(anchors)}
        return loss, metrics
    
    def _get_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distances.
        
        Args:
            embeddings: Feature vectors of shape (batch_size, embedding_dim)
            
        Returns:
            Distance matrix of shape (batch_size, batch_size)
        """
        # Get squared L2 norm for each embedding
        square_sum = torch.sum(embeddings ** 2, dim=1, keepdim=True)
        
        # Compute squared distance matrix using broadcasting
        # dist_mat = (x-y)^2 = x^2 - 2xy + y^2
        dist_mat = square_sum + square_sum.t() - 2.0 * torch.matmul(embeddings, embeddings.t())
        
        # Ensure diagonal is zero and everything is positive
        dist_mat = F.relu(dist_mat)
        dist_mat.fill_diagonal_(0)
        
        return dist_mat
    
    def _get_all_triplets(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate all possible triplets.
        
        Args:
            labels: Identity labels of shape (batch_size)
            
        Returns:
            Tuple of (anchors, positives, negatives) indices
        """
        device = labels.device
        anchors, positives, negatives = [], [], []
        
        # For each anchor
        for i, label_i in enumerate(labels):
            # Find all positives
            pos_indices = torch.where(labels == label_i)[0]
            pos_indices = pos_indices[pos_indices != i]  # Exclude the anchor itself
            
            # Skip if no positives
            if len(pos_indices) == 0:
                continue
                
            # Find all negatives
            neg_indices = torch.where(labels != label_i)[0]
            
            # Skip if no negatives
            if len(neg_indices) == 0:
                continue
                
            # Combine all positives with all negatives
            for p in pos_indices:
                for n in neg_indices:
                    anchors.append(i)
                    positives.append(p.item())
                    negatives.append(n.item())
                    
        if not anchors:
            return tuple([torch.tensor([]) for _ in range(3)])
            
        return (torch.tensor(anchors, device=device), 
                torch.tensor(positives, device=device), 
                torch.tensor(negatives, device=device))
    
    def _get_mined_triplets(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine triplets according to the selected strategy.
        
        Args:
            dist_mat: Distance matrix of shape (batch_size, batch_size)
            labels: Identity labels of shape (batch_size)
            
        Returns:
            Tuple of (anchors, positives, negatives) indices
        """
        device = labels.device
        anchors, positives, negatives = [], [], []
        
        # For each anchor
        for i, label_i in enumerate(labels):
            # Find all positives
            pos_indices = torch.where(labels == label_i)[0]
            pos_indices = pos_indices[pos_indices != i]  # Exclude the anchor itself
            
            # Skip if no positives
            if len(pos_indices) == 0:
                continue
                
            # Find all negatives
            neg_indices = torch.where(labels != label_i)[0]
            
            # Skip if no negatives
            if len(neg_indices) == 0:
                continue
                
            # Get distances to all negatives for this anchor
            anchor_neg_dists = dist_mat[i, neg_indices]
            
            # Select mining strategy
            for p in pos_indices:
                pos_dist = dist_mat[i, p]
                
                if self.mining_strategy == 'hard':
                    # Select hardest negative (closest to anchor)
                    neg_idx = neg_indices[torch.argmin(anchor_neg_dists)]
                    
                    # Only include if this forms a valid triplet
                    neg_dist = dist_mat[i, neg_idx]
                    if neg_dist < pos_dist:
                        anchors.append(i)
                        positives.append(p.item())
                        negatives.append(neg_idx.item())
                
                elif self.mining_strategy == 'semi-hard':
                    # Select semi-hard negatives: further than positive but within margin
                    # D(a,p) < D(a,n) < D(a,p) + margin
                    mask = (anchor_neg_dists > pos_dist) & (anchor_neg_dists < pos_dist + self.margin)
                    
                    if torch.sum(mask) > 0:
                        # Randomly select one semi-hard negative
                        semi_hard_negs = neg_indices[mask]
                        neg_idx = semi_hard_negs[torch.randint(0, len(semi_hard_negs), (1,), device=device)]
                        
                        anchors.append(i)
                        positives.append(p.item())
                        negatives.append(neg_idx.item())
                    else:
                        # Fallback to the hardest negative if no semi-hard found
                        neg_idx = neg_indices[torch.argmin(anchor_neg_dists)]
                        anchors.append(i)
                        positives.append(p.item())
                        negatives.append(neg_idx.item())
        
        if not anchors:
            return tuple([torch.tensor([]) for _ in range(3)])
            
        return (torch.tensor(anchors, device=device), 
                torch.tensor(positives, device=device), 
                torch.tensor(negatives, device=device))

    def _compute_triplet_loss(self, dist_mat: torch.Tensor, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss for the selected triplets.
        
        Args:
            dist_mat: Distance matrix of shape (batch_size, batch_size)
            anchors: Indices of anchor samples
            positives: Indices of positive samples
            negatives: Indices of negative samples
            
        Returns:
            Scalar loss value
        """
        anchor_pos_dist = dist_mat[anchors, positives]
        anchor_neg_dist = dist_mat[anchors, negatives]
        
        # Triplet loss: max(0, D(a,p) - D(a,n) + margin)
        triplet_loss = F.relu(anchor_pos_dist - anchor_neg_dist + self.margin)
        
        return triplet_loss.mean()


class CenterLoss(nn.Module):
    """
    Center loss for face recognition.
    
    Center loss minimizes the intra-class variations by penalizing the distances
    between the features and their corresponding class centers.
    """
    
    def __init__(self, num_classes: int, feature_dim: int, alpha: float = 0.5):
        """
        Initialize center loss.
        
        Args:
            num_classes: Number of identity classes
            feature_dim: Dimension of the feature vector
            alpha: Learning rate for centers update
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        # Initialize class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss.
        
        Args:
            features: Feature vectors of shape (batch_size, feature_dim)
            labels: Identity labels of shape (batch_size)
            
        Returns:
            Loss value
        """
        # Gather centers for each sample's label
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        
        # Compute distances to centers
        diff = features - centers_batch
        
        # Compute squared distances and take mean
        loss = torch.sum(torch.pow(diff, 2)) / batch_size
        
        return loss
        
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update class centers.
        
        Args:
            features: Feature vectors of shape (batch_size, feature_dim)
            labels: Identity labels of shape (batch_size)
        """
        # Manually update centers (should be called after optimizer.step())
        # For each class, compute mean of features and update center
        for i in range(self.num_classes):
            mask = labels == i
            if torch.sum(mask) > 0:
                class_mean = torch.mean(features[mask], dim=0)
                self.centers.data[i] = self.centers.data[i] - self.alpha * (self.centers.data[i] - class_mean)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss.
    
    This is a self-supervised learning loss that encourages embeddings of the same class 
    to be close and those of different classes to be far apart, while considering multiple
    positive examples per anchor.
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07, contrast_mode: str = 'all'):
        """
        Initialize supervised contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling
            base_temperature: Base temperature (typically the same as temperature)
            contrast_mode: 'all' or 'one' - determines whether to include multiple positives per anchor
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature vectors, shape [batch_size, n_views, feature_dim] or [batch_size, feature_dim]
            labels: Class labels, shape [batch_size]
            mask: Optional binary mask for valid pairs, shape [batch_size, batch_size]
            
        Returns:
            Loss value
        """
        device = features.device
        
        # Handle single view case: reshape to [batch_size, 1, feature_dim]
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        
        # If labels not provided, create pseudo-labels based on instance identity
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            # Create mask based on class labels
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Get normalized features
        features = F.normalize(features, p=2, dim=-1)
        
        # Compute similarity matrix
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            # Only use the first view for anchors
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            # Use all views as anchors
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # Mask out self-contrast cases for log_softmax
        logits_mask = torch.scatter(
            torch.ones_like(anchor_dot_contrast),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )
        
        # Create positive mask by extending original mask according to # of views
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask out self-contrast cases
        logits_mask = logits_mask * ((1 - torch.eye(batch_size * anchor_count, device=device)) @ torch.ones(
            (batch_size * anchor_count, batch_size * contrast_count), device=device))
        
        mask = mask * logits_mask
        
        # Compute log-probabilities
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Scale by temperature
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class CircleLoss(nn.Module):
    """
    Circle loss for face recognition.
    
    Circle loss unifies the triplet and softmax-based losses with a unified distance margin.
    It provides a more flexible optimization approach for similarity learning.
    """
    
    def __init__(self, margin: float = 0.25, gamma: float = 256):
        """
        Initialize circle loss.
        
        Args:
            margin: Margin for separation
            gamma: Scale factor for similarity scores
        """
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute circle loss.
        
        Args:
            embeddings: Feature vectors of shape (batch_size, feature_dim)
            labels: Identity labels of shape (batch_size)
            
        Returns:
            Loss value
        """
        # Get cosine similarity matrix
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_mat = torch.matmul(embeddings, embeddings.t())
        
        # Create mask for positive and negative pairs
        label_mat = labels.expand(labels.size(0), labels.size(0))
        pos_mask = (label_mat == label_mat.t()).float()
        neg_mask = 1.0 - pos_mask
        
        # Remove diagonal elements 
        pos_mask.fill_diagonal_(0)
        
        # Get positive and negative similarities
        pos_sim = sim_mat * pos_mask
        neg_sim = sim_mat * neg_mask
        
        # For numerical stability
        pos_sim_exp = torch.exp(-self.gamma * (pos_sim - (1 - self.margin)))
        neg_sim_exp = torch.exp(self.gamma * (neg_sim - self.margin))
        
        # Sum over positives and negatives
        num_pos = pos_mask.sum(dim=1, keepdim=True)
        num_neg = neg_mask.sum(dim=1, keepdim=True)
        
        # Compute final loss
        loss = torch.log(1 + 
                (pos_sim_exp.sum(dim=1, keepdim=True) / (num_pos + 1e-8)) * 
                (neg_sim_exp.sum(dim=1, keepdim=True) / (num_neg + 1e-8))
               ).mean()
        
        return loss 


class TripletLoss(nn.Module):
    """
    Basic triplet loss for pre-generated triplets.
    
    This class implements the standard triplet loss formula for triplets
    passed directly as (anchor, positive, negative) embeddings.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize the triplet loss.
        
        Args:
            margin: Margin to enforce between positive and negative distances
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss for pre-generated triplets.
        
        Args:
            anchor: Anchor embeddings of shape (batch_size, embedding_dim)
            positive: Positive embeddings of shape (batch_size, embedding_dim)
            negative: Negative embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            Scalar loss value
        """
        # Compute distances
        pos_dist = torch.sum((anchor - positive).pow(2), dim=1)
        neg_dist = torch.sum((anchor - negative).pow(2), dim=1)
        
        # Compute triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Return mean loss
        return losses.mean() 