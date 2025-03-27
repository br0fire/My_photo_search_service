import os
import json
import random
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class FaceImageDataset(Dataset):
    """
    Dataset for face images from Roboflow.
    
    This dataset handles the face recognition data downloaded from Roboflow,
    adapting it for training a Siamese network.
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the face image dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Data split to use ('train', 'valid', or 'test')
            transform: Optional transform to apply to the images
            target_size: Size to resize the images to
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Define standard paths based on Roboflow folder structure
        self.split_dir = os.path.join(data_dir, split)
        self.images_dir = os.path.join(self.split_dir, 'images')
        
        # Load images and annotations
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.images_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
            
        # Extract identity information from filenames or annotations
        self.identities, self.identity_to_idx = self._extract_identities()
        
        # Map each image to its identity
        self.image_identities = self._map_images_to_identities()
        
        print(f"Loaded {len(self.image_files)} images with {len(self.identities)} identities")
        
    def _extract_identities(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Extract unique identity labels from the dataset.
        
        This method extracts identity information from filenames or annotation files.
        
        Returns:
            A tuple containing:
                - List of unique identity strings
                - Dictionary mapping identity strings to integer indices
        """
        # In a real dataset, we would extract identities from annotations
        # Here we'll assume identities are encoded in filenames (e.g., person_1_image_2.jpg)
        
        identities = set()
        
        # Look for identity in filename (assumes format like "person_id_*.jpg")
        for img_file in self.image_files:
            # Extract identity from filename - adapt this based on your naming convention
            parts = os.path.splitext(img_file)[0].split('_')
            
            # Try to infer identity from filename
            if len(parts) >= 2:
                # Assume format is like "person_id" or "id_image"
                identity = parts[0]  # Modify this based on your naming convention
                identities.add(identity)
        
        # If no identities found, check for JSON annotations
        if len(identities) <= 1:
            # Try to load from a metadata file if it exists
            metadata_file = os.path.join(self.split_dir, 'metadata.json')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract identities from metadata (format will depend on your data)
                    if isinstance(metadata, dict) and 'identities' in metadata:
                        identities = set(metadata['identities'])
                    elif isinstance(metadata, list):
                        # Assume it's a list of image data with identity info
                        for item in metadata:
                            if isinstance(item, dict) and 'identity' in item:
                                identities.add(item['identity'])
                except:
                    pass
        
        # If still no identities found, use the whole filename as identity (not ideal)
        if len(identities) <= 1:
            print("Warning: Could not extract identities from filenames or metadata.")
            print("Using filenames as identity (not ideal for real face recognition).")
            identities = set([os.path.splitext(f)[0] for f in self.image_files])
        
        # Convert to list and create mapping to indices
        identities_list = sorted(list(identities))
        identity_to_idx = {identity: idx for idx, identity in enumerate(identities_list)}
        
        return identities_list, identity_to_idx
    
    def _map_images_to_identities(self) -> List[int]:
        """
        Map each image to its identity index.
        
        Returns:
            List of identity indices for each image
        """
        image_identities = []
        
        # First try to use filenames
        for img_file in self.image_files:
            parts = os.path.splitext(img_file)[0].split('_')
            
            if len(parts) >= 2:
                # Assume format is like "person_id" or "id_image"
                identity = parts[0]  # Modify based on your naming convention
                if identity in self.identity_to_idx:
                    image_identities.append(self.identity_to_idx[identity])
                    continue
            
            # Fallback to using the whole filename
            identity = os.path.splitext(img_file)[0]
            image_identities.append(self.identity_to_idx.get(identity, 0))
            
        # Check if we have separate annotation files
        # (code to parse annotation files would go here if needed)
        
        return image_identities
        
    def __len__(self) -> int:
        """
        Get the number of images in the dataset.
        
        Returns:
            Number of images
        """
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its corresponding identity index.
        
        Args:
            idx: Index of the image to retrieve
            
        Returns:
            Tuple of (image_tensor, identity_idx)
        """
        img_file = self.image_files[idx]
        identity_idx = self.image_identities[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # Resize image
        image = image.resize(self.target_size, Image.BILINEAR)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = F.to_tensor(image)
            image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return image, identity_idx
    
    def get_identity_images(self, identity_idx: int) -> List[int]:
        """
        Get all image indices for a given identity.
        
        Args:
            identity_idx: Index of the identity
            
        Returns:
            List of image indices belonging to the identity
        """
        return [i for i, identity in enumerate(self.image_identities) if identity == identity_idx]


class FaceTripletDataset(Dataset):
    """
    Dataset for generating triplets for face recognition training.
    
    This dataset creates triplets from a base dataset, where each triplet consists of:
    - An anchor image
    - A positive image (same identity as anchor)
    - A negative image (different identity from anchor)
    """
    
    def __init__(self, 
                 base_dataset: FaceImageDataset,
                 num_triplets: int = 10000,
                 triplet_type: str = 'random'):
        """
        Initialize the triplet dataset.
        
        Args:
            base_dataset: Base dataset containing face images
            num_triplets: Number of triplets to generate
            triplet_type: Type of triplet selection ('random', 'hard', or 'semi-hard')
        """
        self.base_dataset = base_dataset
        self.num_triplets = num_triplets
        self.triplet_type = triplet_type
        
        # Create identity-based indices for efficient triplet generation
        self.identity_images = {}
        for identity_idx in range(len(base_dataset.identities)):
            self.identity_images[identity_idx] = base_dataset.get_identity_images(identity_idx)
            
        # Generate triplets
        self.triplets = self._generate_triplets()
        
    def _generate_triplets(self) -> List[Tuple[int, int, int]]:
        """
        Generate triplets for training.
        
        Returns:
            List of (anchor_idx, positive_idx, negative_idx) triplets
        """
        triplets = []
        
        # Filter out identities with less than 2 images (need at least 2 for positive pair)
        valid_identities = [idx for idx, images in self.identity_images.items() if len(images) >= 2]
        
        if not valid_identities:
            raise ValueError("No valid identities found for triplet generation. " +
                            "Each identity must have at least 2 images.")
        
        # For random triplets, just randomly select (A,P,N)
        for _ in range(self.num_triplets):
            # Select a random identity for anchor/positive
            anchor_identity = random.choice(valid_identities)
            
            # Select a different identity for negative
            other_identities = [idx for idx in valid_identities if idx != anchor_identity]
            if not other_identities:
                continue  # Skip if only one identity
            
            negative_identity = random.choice(other_identities)
            
            # Select two different images for anchor and positive
            if len(self.identity_images[anchor_identity]) < 2:
                continue
                
            anchor_idx, positive_idx = random.sample(self.identity_images[anchor_identity], 2)
            
            # Select a random image for negative
            negative_idx = random.choice(self.identity_images[negative_identity])
            
            triplets.append((anchor_idx, positive_idx, negative_idx))
            
        return triplets
    
    def __len__(self) -> int:
        """
        Get the number of triplets in the dataset.
        
        Returns:
            Number of triplets
        """
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Get a triplet of images.
        
        Args:
            idx: Index of the triplet to retrieve
            
        Returns:
            Tuple of (anchor_image, positive_image, negative_image, [anchor_identity, positive_identity, negative_identity])
        """
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        
        # Get images and identities
        anchor_img, anchor_identity = self.base_dataset[anchor_idx]
        positive_img, positive_identity = self.base_dataset[positive_idx]
        negative_img, negative_identity = self.base_dataset[negative_idx]
        
        return anchor_img, positive_img, negative_img, [anchor_identity, positive_identity, negative_identity]


class FacePairDataset(Dataset):
    """
    Dataset for generating positive and negative pairs for face recognition training.
    
    This dataset creates pairs from a base dataset, where each pair consists of:
    - An anchor image
    - A second image (either positive or negative)
    - A label (1 for positive, 0 for negative)
    """
    
    def __init__(self, 
                 base_dataset: FaceImageDataset,
                 num_pairs: int = 10000,
                 pos_neg_ratio: float = 0.5):
        """
        Initialize the pair dataset.
        
        Args:
            base_dataset: Base dataset containing face images
            num_pairs: Number of pairs to generate
            pos_neg_ratio: Ratio of positive pairs to total pairs (0.5 = balanced)
        """
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        self.pos_neg_ratio = pos_neg_ratio
        
        # Create identity-based indices for efficient pair generation
        self.identity_images = {}
        for identity_idx in range(len(base_dataset.identities)):
            self.identity_images[identity_idx] = base_dataset.get_identity_images(identity_idx)
            
        # Generate pairs
        self.pairs = self._generate_pairs()
        
    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Generate pairs for training.
        
        Returns:
            List of (first_idx, second_idx, label) pairs where label is 1 for positive, 0 for negative
        """
        pairs = []
        
        # Calculate number of positive and negative pairs
        num_positive = int(self.num_pairs * self.pos_neg_ratio)
        num_negative = self.num_pairs - num_positive
        
        # Filter out identities with less than 2 images (need at least 2 for positive pair)
        valid_identities = [idx for idx, images in self.identity_images.items() if len(images) >= 2]
        
        if not valid_identities:
            raise ValueError("No valid identities found for pair generation. " +
                            "Each identity must have at least 2 images.")
        
        # Generate positive pairs (same identity)
        for _ in range(num_positive):
            # Select a random identity with at least 2 images
            identity_idx = random.choice(valid_identities)
            
            # Select two different images for the pair
            img1_idx, img2_idx = random.sample(self.identity_images[identity_idx], 2)
            
            pairs.append((img1_idx, img2_idx, 1))  # 1 indicates positive pair
            
        # Generate negative pairs (different identities)
        for _ in range(num_negative):
            # Select two different identities
            if len(valid_identities) < 2:
                # If only one valid identity, we can't create negative pairs
                continue
                
            identity1_idx, identity2_idx = random.sample(valid_identities, 2)
            
            # Select one image from each identity
            img1_idx = random.choice(self.identity_images[identity1_idx])
            img2_idx = random.choice(self.identity_images[identity2_idx])
            
            pairs.append((img1_idx, img2_idx, 0))  # 0 indicates negative pair
            
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def __len__(self) -> int:
        """
        Get the number of pairs in the dataset.
        
        Returns:
            Number of pairs
        """
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of images with label.
        
        Args:
            idx: Index of the pair to retrieve
            
        Returns:
            Tuple of (first_image, second_image, label_tensor)
        """
        img1_idx, img2_idx, label = self.pairs[idx]
        
        # Get images
        img1, _ = self.base_dataset[img1_idx]
        img2, _ = self.base_dataset[img2_idx]
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_face_data_loaders(
    data_dir: str,
    mode: str = 'triplet',
    batch_size: int = 32,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    num_pairs_or_triplets: int = 10000,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create data loaders for face recognition.
    
    Args:
        data_dir: Directory containing the dataset
        mode: Training mode ('triplet' or 'pair')
        batch_size: Batch size for data loaders
        train_transform: Optional transforms for training set
        val_transform: Optional transforms for validation set
        num_pairs_or_triplets: Number of pairs or triplets to generate
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary of data loaders for 'train', 'val', and 'test'
    """
    # Define default transforms if none provided
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    # Create base datasets
    train_dataset = FaceImageDataset(data_dir, split='train', transform=train_transform)
    val_dataset = FaceImageDataset(data_dir, split='valid', transform=val_transform)
    test_dataset = FaceImageDataset(data_dir, split='test', transform=val_transform)
    
    # Create appropriate datasets based on mode
    if mode == 'triplet':
        train_triplet_dataset = FaceTripletDataset(train_dataset, num_triplets=num_pairs_or_triplets)
        val_triplet_dataset = FaceTripletDataset(val_dataset, num_triplets=num_pairs_or_triplets // 2)
        test_triplet_dataset = FaceTripletDataset(test_dataset, num_triplets=num_pairs_or_triplets // 2)
        
        train_loader = DataLoader(
            train_triplet_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_triplet_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_triplet_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
    elif mode == 'pair':
        train_pair_dataset = FacePairDataset(train_dataset, num_pairs=num_pairs_or_triplets)
        val_pair_dataset = FacePairDataset(val_dataset, num_pairs=num_pairs_or_triplets // 2)
        test_pair_dataset = FacePairDataset(test_dataset, num_pairs=num_pairs_or_triplets // 2)
        
        train_loader = DataLoader(
            train_pair_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_pair_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_pair_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
    else:
        # Regular classification mode
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 