# Siamese Triplet Network for Face Recognition

This folder contains the implementation of a face recognition system using Siamese networks with triplet loss.

## Features

- Multiple model architectures (ResNet, EfficientNet, MobileNetV2, etc.)
- Training scripts with various loss functions and mining strategies
- Evaluation tools for model performance
- Face identification script for new images

## Getting Started

See the [QUICKSTART.md](QUICKSTART.md) for getting started quickly and [TRAINING.md](TRAINING.md) for more detailed training instructions.

## Scripts

- `face_recognition.py`: Consolidated model and evaluation functionality
- `evaluate_face_model.sh`: Script for running evaluations
- `train_triplet_network.py`: Script for training with triplet loss
- `identify_face.py`: Script for recognizing faces in new images
- `run_quick_training.sh` and `run_recommended_training.sh`: Automated training pipelines
