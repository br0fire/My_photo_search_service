# Face Recognition System
This repository contains a face recognition system based on Siamese networks with triplet loss, implemented in PyTorch.

## Features

- High-performance face embedding models
- Evaluation tools for model performance
- Optimized for real-world use with challenging images

## Directory Structure

- `face_recognition/`: Consolidated implementation with model and evaluation code
  - `face_recognition.py`: Main script with model architecture and evaluation functionality
  - `evaluate_face_model.sh`: Convenient script for running evaluations

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
cd face_recognition
./evaluate_face_model.sh --model path/to/model.pth --test-dir path/to/test/dir
```

See the README in the face_recognition directory for more detailed instructions.
