#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Face Identification Script

This script loads a trained face recognition model and uses it to identify faces in new images
by comparing them with a reference dataset. It finds the closest matches and displays the results.

Usage:
    python identify_face.py --model models/resnet50/final_model.pth --image path/to/test_image.jpg

Author: User
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

# Add the parent directory to the path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from pretrained_face_networks import PretrainedFaceEmbedding, PretrainedFaceEmbeddingWithL2, FaceTripletNet
from face_datasets import FaceImageDataset


def load_model(model_path, device):
    """
    Load a trained face recognition model.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        device (torch.device): Device to load the model on
    
    Returns:
        model: Loaded face recognition model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model_name = checkpoint.get('model_name', 'resnet50')
    embedding_dim = checkpoint.get('embedding_dim', 128)
    freeze_backbone = checkpoint.get('freeze_backbone', False)
    
    # Create the same model architecture
    embedding_net = PretrainedFaceEmbeddingWithL2(
        embedding_dim=embedding_dim,
        model_name=model_name,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    model = FaceTripletNet(embedding_net)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def recognize_face(model, test_img_path, reference_dataset, top_k=5, threshold=0.5, device=None):
    """
    Recognize a face by finding the closest matches in the reference dataset.
    
    Args:
        model: Trained face recognition model
        test_img_path (str): Path to the test image
        reference_dataset: Dataset containing reference face images
        top_k (int): Number of top matches to return
        threshold (float): Distance threshold for match/no match decision
        device (torch.device): Device to run inference on
    
    Returns:
        dict: Dictionary containing match information
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load and preprocess the test image
    test_img = Image.open(test_img_path).convert('RGB')
    
    # Define image transformation
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_tensor = preprocess(test_img).unsqueeze(0).to(device)
    
    # Get the embedding for the test image
    with torch.no_grad():
        test_embedding = model.get_embedding(test_tensor)
    
    # Compute distances to all images in the reference dataset
    print(f"Computing distances to {len(reference_dataset)} reference images...")
    distances = []
    identities = []
    img_paths = []
    
    for i in range(len(reference_dataset)):
        img, identity = reference_dataset[i]
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            ref_embedding = model.get_embedding(img)
            
        # Compute Euclidean distance
        dist = torch.sum((test_embedding - ref_embedding) ** 2).item()
        distances.append(dist)
        identities.append(identity)
        img_paths.append(reference_dataset.image_files[i])
    
    # Sort by distance (ascending)
    sorted_indices = np.argsort(distances)
    sorted_distances = [distances[i] for i in sorted_indices[:top_k]]
    sorted_identities = [identities[i] for i in sorted_indices[:top_k]]
    sorted_img_paths = [img_paths[i] for i in sorted_indices[:top_k]]
    
    # Map identity indices to names
    identity_names = [reference_dataset.identities[id_idx] for id_idx in sorted_identities]
    
    # Check if any match is below threshold (a valid match)
    valid_match = any(dist < threshold for dist in sorted_distances)
    
    # Visualize the results
    plt.figure(figsize=(15, 5))
    
    # Display the test image
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(test_img)
    plt.title("Query Image")
    plt.axis('off')
    
    # Display the top matches
    for i in range(top_k):
        plt.subplot(1, top_k + 1, i + 2)
        match_img_path = os.path.join(reference_dataset.images_dir, sorted_img_paths[i])
        match_img = Image.open(match_img_path).convert('RGB')
        plt.imshow(match_img)
        
        # Color-code based on threshold
        color = 'green' if sorted_distances[i] < threshold else 'red'
        match_status = 'MATCH' if sorted_distances[i] < threshold else 'NO MATCH'
        
        plt.title(f"{match_status}\n{identity_names[i]}\nDist: {sorted_distances[i]:.4f}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('recognition_result.png')
    plt.show()
    
    print("\nTop matches:")
    for i in range(top_k):
        match_status = 'MATCH ✓' if sorted_distances[i] < threshold else 'NO MATCH ✗'
        print(f"{i+1}. {match_status} - {identity_names[i]} (Distance: {sorted_distances[i]:.4f})")
    
    # Return match information
    return {
        'distances': sorted_distances,
        'identities': sorted_identities,
        'identity_names': identity_names,
        'image_paths': sorted_img_paths,
        'valid_match': valid_match,
        'best_match': identity_names[0] if valid_match and sorted_distances[0] < threshold else None
    }


def main():
    """Main function for the face identification script."""
    parser = argparse.ArgumentParser(description='Face Identification Script')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to the test image')
    parser.add_argument('--data-dir', type=str, default='data', help='Path to the reference dataset')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top matches to display')
    parser.add_argument('--threshold', type=float, default=0.6, help='Distance threshold for match/no match decision')
    args = parser.parse_args()
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    
    # Load the reference dataset
    print(f"Loading reference dataset from {args.data_dir}...")
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset object (using validation set as reference)
    reference_dataset = FaceImageDataset(args.data_dir, 'valid', transform, target_size=(224, 224))
    print(f"Reference dataset: {len(reference_dataset)} images, {len(reference_dataset.identities)} identities")
    
    # Recognize the face
    print(f"Recognizing face in {args.image}...")
    results = recognize_face(
        model, 
        args.image, 
        reference_dataset, 
        top_k=args.top_k, 
        threshold=args.threshold,
        device=device
    )
    
    # Output the best match
    if results['valid_match']:
        print(f"\nBest match: {results['best_match']} (Distance: {results['distances'][0]:.4f})")
    else:
        print("\nNo valid match found. The person may not be in the reference dataset.")


if __name__ == '__main__':
    main() 