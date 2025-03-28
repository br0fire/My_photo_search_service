#!/usr/bin/env python3
"""
Script to download the My Photo Search dataset from Roboflow.

This script downloads the face recognition dataset from Roboflow and organizes it
for training a Siamese/Triplet network.
"""

import os
import argparse
from typing import Optional, Union, Any
from pathlib import Path
from roboflow import Roboflow


def download_dataset(
    api_key: str,
    workspace: str = "melnikum",
    project: str = "my-photo-search-2", 
    version: int = 2,
    output_dir: str = "data",
    format: str = "folder"
) -> str:
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version
        output_dir: Directory to save the dataset
        format: Download format (e.g., 'folder', 'coco')
        
    Returns:
        Path to the downloaded dataset
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get the project
    project = rf.workspace(workspace).project(project)
    
    # Get the version
    version = project.version(version)
    
    # Download the dataset
    print(f"Downloading dataset to {output_dir}...")
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset
    dataset = version.download(format, location=output_dir)
    
    # The Roboflow API might download to a subdirectory within output_dir
    # Check if the dataset object has an attribute with the download path
    dataset_path = output_dir
    if hasattr(dataset, 'location'):
        dataset_path = dataset.location
    
    # Extract the actual download location
    # If the path doesn't exist, try to find where the data is actually stored
    if not os.path.exists(os.path.join(dataset_path, 'train')):
        # Look for possible dataset locations within the output directory
        potential_paths = [
            os.path.join(output_dir, 'train'),
            os.path.join(output_dir, project.name),
            os.path.join(output_dir, f"{project.name}-{version.version}")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                dataset_path = os.path.dirname(path)
                break
    
    print(f"Dataset downloaded. Using path: {dataset_path}")
    return dataset_path


def create_identity_metadata(dataset_path: str, output_file: Optional[str] = None) -> None:
    """
    Create metadata file with identity information.
    
    Args:
        dataset_path: Path to the downloaded dataset
        output_file: Path to save the metadata file (if None, will use 'metadata.json' in each split dir)
    """
    import json
    
    # Ensure dataset path exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path {dataset_path} does not exist.")
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"Split directory {split_dir} not found, skipping.")
            continue
        
        # Check directory structure - Roboflow might have organized by class folders
        # instead of having an 'images' subdirectory
        images_dir = split_dir / 'images'
        class_based_structure = not images_dir.exists() and any(
            f.is_dir() for f in split_dir.iterdir()
        )
        
        identities = set()
        image_identities = {}
        
        if class_based_structure:
            print(f"Found class-based directory structure in {split_dir}")
            # In this structure, each class/identity has its own directory
            for identity_dir in [d for d in split_dir.iterdir() if d.is_dir()]:
                identity = identity_dir.name
                identities.add(identity)
                
                # Get all images in this identity directory
                for img_file in [f for f in identity_dir.iterdir() 
                                if f.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]:
                    # Store with path relative to split directory for consistency
                    rel_path = os.path.join(identity, img_file.name)
                    image_identities[rel_path] = identity
            
            # Create images directory to satisfy FaceImageDataset expectations
            if not images_dir.exists():
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # Create symlinks to original images
                for rel_path in image_identities:
                    identity, img_name = os.path.split(rel_path)
                    # Create a filename that preserves identity information
                    target_filename = f"{identity}_{img_name}"
                    source_path = split_dir / rel_path
                    target_path = images_dir / target_filename
                    
                    # Copy the file instead of symlink for better compatibility
                    import shutil
                    shutil.copy2(source_path, target_path)
                    
                print(f"Created {images_dir} with symlinks to original images")
        else:
            if not images_dir.exists():
                print(f"Images directory {images_dir} not found, skipping.")
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(images_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # This depends on your naming convention - adjust as needed
            for img_file in image_files:
                # Extract identity from filename - adjust this based on your naming convention
                parts = os.path.splitext(img_file)[0].split('_')
                
                if len(parts) >= 2:
                    # Assume format is like "person_id" or "id_image"
                    identity = parts[0]  # Modify this based on your naming convention
                else:
                    # Fallback to whole filename
                    identity = os.path.splitext(img_file)[0]
                    
                identities.add(identity)
                image_identities[img_file] = identity
        
        # Create metadata
        metadata = {
            'identities': list(identities),
            'image_identities': image_identities
        }
        
        # Determine output file
        if output_file is None:
            metadata_file = split_dir / 'metadata.json'
        else:
            metadata_file = Path(output_file).with_suffix('.json')
            if not metadata_file.parent.exists():
                metadata_file.parent.mkdir(parents=True)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Created metadata file {metadata_file} with {len(identities)} identities.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download Roboflow dataset for face recognition")
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--workspace", type=str, default="melnikum", help="Roboflow workspace name")
    parser.add_argument("--project", type=str, default="my-photo-search-2", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=2, help="Dataset version")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save the dataset")
    parser.add_argument("--format", type=str, default="folder", help="Download format")
    
    args = parser.parse_args()
    
    # Download dataset
    dataset_path = download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        output_dir=args.output_dir,
        format=args.format
    )
    
    # Create metadata file with identity information
    create_identity_metadata(dataset_path)


if __name__ == "__main__":
    main() 