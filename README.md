# My_photo_search_service

This repository contains an open-source deep learning system designed for robust person recognition in large-scale photo libraries, specifically tailored for practical, real-world use in event-based image retrieval scenarios.

## Overview

As personal photo libraries grow exponentially, manually searching for images featuring specific individuals becomes increasingly difficult. This project provides an efficient automated solution integrating state-of-the-art face detection and recognition models optimized for complex, real-world conditions.

## Key Features

- **High Accuracy**: Combines YOLOv8 face detection with custom-trained FaceTripleNet and FaceNet embeddings.
- **Real-world Performance**: Optimized to handle typical photo collection challenges such as varying lighting conditions, occlusions, and crowded scenes.
- **Fast Processing**: Efficient inference pipeline ensuring usability even on large datasets.
- **Open-Source**: Fully open-source, enabling further customization and improvement.

## Methodology

Our system employs a pipeline consisting of:
1. **Face Detection**: Using YOLOv8 for speed and accuracy.
2. **Face Recognition**: Generating embeddings using FaceTripleNet and FaceNet.
3. **Retrieval**: Nearest-neighbor search using distance function.

## Dataset

A custom, labeled dataset created from public events (AIRI meetups and Innovation Workshop at Skoltech) was used for training and evaluation. It includes:

- Total images collected: ~6000
- Final curated dataset: 2449 high-quality, labeled face images
- Target person images: 49
- Non-target images: 1404
- Invalid detections: 987

The dataset addresses typical event photography challenges (varying poses, illumination conditions, occlusion).

### [Download Labeled Train Dataset](https://app.roboflow.com/melnikum/my-photo-search-2/browse)
### [Download Labeled Test Dataset](https://app.roboflow.com/melnikum/my-photo-search-2/browse)

## Results

Our system outperforms baseline models, including pre-trained ArcFace and VGG-Face combined with YOLOv8 detection:

| Model | F1 Score | Recall | Precision |
|-------|----------|--------|-----------|
| YOLOv8 + ArcFace | 0.58 | 0.78 | 0.46 |
| YOLOv8 + VGG-Face | 0.73 | 0.82 | 0.66 |
| **FaceTripleNet (ours)** | **0.78** | **0.85** | 0.72 |
| **FaceNet (ours)** | 0.74 | 0.60 | **0.92** |

## Repository Structure

```
- detection/
  - detect.py
- preprocess/
  deblurring.py
  deduplicate.py
  unzip.sh
- recognition/
  - ...
  - ...
- evaluation/
  - DeepFace benchmark.ipynb
- data/
- README.md
- requirements.txt
- slides.pdf
```

## Authors

- Ivan Listopadov
- Sergey Grozny
- Alexander Zaytsev
- Yurii Melnik
- Petr Sokerin

