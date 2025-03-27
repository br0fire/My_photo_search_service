# Face Recognition Training Guide

This guide provides instructions for training Siamese networks for face recognition using triplet loss. The implementation supports various pretrained backbones and different mining strategies to achieve optimal results.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Basic Training](#basic-training)
4. [Advanced Training Options](#advanced-training-options)
5. [Evaluation](#evaluation)
6. [Experiment Tracking](#experiment-tracking)
7. [Inference](#inference)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before starting, ensure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have a CUDA-capable GPU for faster training. If not, training will automatically fall back to CPU, but will be significantly slower.

## Data Preparation

### 1. Download Dataset

First, download the face dataset from Roboflow using the provided script:

```bash
python download_roboflow_data.py --api-key="YOUR_ROBOFLOW_API_KEY" --output-dir="data"
```

This script downloads the dataset and organizes it into the following structure:

```
data/
  ├── train/
  │   ├── images/
  │   └── metadata.json
  ├── valid/
  │   ├── images/
  │   └── metadata.json
  └── test/
      ├── images/
      └── metadata.json
```

### 2. Verify Dataset

Ensure that your dataset has been properly organized and contains multiple images per identity for triplet creation:

```bash
python -c "from face_datasets import FaceImageDataset; ds = FaceImageDataset('data', 'train'); print(f'Train: {len(ds)} images, {len(ds.identities)} identities')"
```

## Basic Training

The simplest way to train a model is to use the provided training script with default parameters:

```bash
python train_triplet_network.py --data-dir data --model resnet50
```

This will train a ResNet-50 model with semi-hard triplet mining for 20 epochs, saving results to the `experiments` directory.

### Quick Start Commands

Here are some common training commands to get started:

1. **Train with ResNet-50 and semi-hard mining (recommended for beginners):**
   ```bash
   python train_triplet_network.py --data-dir data --model resnet50 --mining-strategy semi-hard
   ```

2. **Train with a smaller, faster model (good for limited hardware):**
   ```bash
   python train_triplet_network.py --data-dir data --model mobilenetv2 --mining-strategy semi-hard
   ```

3. **Use a higher learning rate with frozen backbone (for quick fine-tuning):**
   ```bash
   python train_triplet_network.py --data-dir data --model resnet50 --freeze-backbone --lr 0.005
   ```

## Advanced Training Options

The training script supports many options to customize your training:

### Model Selection

Choose from various pretrained models:

```bash
# Use a custom CNN (defined in face_networks.py)
python train_triplet_network.py --model custom

# Use ResNet-18 (faster but less accurate)
python train_triplet_network.py --model resnet18

# Use ResNet-50 (good balance of speed and accuracy)
python train_triplet_network.py --model resnet50

# Use EfficientNet-B0 (efficient architecture)
python train_triplet_network.py --model efficientnet_b0

# Use MobileNetV2 (very efficient, good for deployment)
python train_triplet_network.py --model mobilenetv2

# Use Vision Transformer (potentially more accurate but slower)
python train_triplet_network.py --model vit_small
```

### Mining Strategies

Select different triplet mining strategies:

```bash
# Random triplets (baseline, fastest)
python train_triplet_network.py --mining-strategy all

# Hard mining (select hardest negatives, can be unstable)
python train_triplet_network.py --mining-strategy hard

# Semi-hard mining (balanced approach, recommended)
python train_triplet_network.py --mining-strategy semi-hard
```

### Training Parameters

Adjust training parameters to optimize performance:

```bash
# Adjust embedding dimension
python train_triplet_network.py --embedding-dim 256

# Adjust margin for triplet loss
python train_triplet_network.py --margin 0.5

# Adjust batch size (reduce if out of memory)
python train_triplet_network.py --batch-size 16

# Adjust learning rate
python train_triplet_network.py --lr 0.0005

# Train for more epochs
python train_triplet_network.py --epochs 50

# Use a different learning rate scheduler
python train_triplet_network.py --scheduler step
```

### Full Example with Multiple Parameters

Here's an example command combining multiple parameters:

```bash
python train_triplet_network.py \
  --data-dir data \
  --model resnet50 \
  --embedding-dim 256 \
  --mining-strategy semi-hard \
  --batch-size 32 \
  --epochs 30 \
  --lr 0.001 \
  --margin 0.3 \
  --scheduler cosine \
  --output-dir experiments \
  --exp-name resnet50_semihardmining_256dim
```

## Evaluation

After training, the script automatically evaluates the model on the test set. Evaluation includes:

1. Face verification metrics (precision-recall curve, accuracy)
2. t-SNE visualization of embeddings
3. Training/validation loss curves

### Standard Evaluation

You can manually evaluate a trained model for face identification:

```bash
# Evaluate a specific trained model on a single image
python identify_face.py --model experiments/resnet50_semihardmining_20220101_120000/final_model.pth --image path/to/your/test_image.jpg
```

### Comprehensive Binary Classification Evaluation

For a more comprehensive evaluation on a face verification task, use the dedicated evaluation script:

```bash
# Run the evaluation script
./evaluate_face_model.sh --model experiments/YOUR_EXPERIMENT_DIR/final_model.pth --test-dir path/to/test/data
```

The test directory should contain two subdirectories:
- `0`: Contains pairs of images of different people
- `1`: Contains pairs of images of the same person

This script performs the following:
1. Loads all images and computes embeddings
2. Measures distances between pairs of embeddings
3. Finds the optimal threshold for same/different face classification
4. Computes detailed metrics:
   - Accuracy, precision, recall, and F1 score
   - ROC curve with AUC (Area Under Curve)
   - Precision-recall curve
   - Distance distributions for same/different faces
5. Saves all metrics as JSON and generates visualization plots

Example usage:
```bash
./evaluate_face_model.sh \
  --model experiments/resnet50_semihardmining_20230415_123456/final_model.pth \
  --test-dir data/verification_test \
  --output-dir evaluation_results/resnet50_verification
```

You can also specify a custom threshold value instead of using the automatically determined optimal threshold:

```bash
./evaluate_face_model.sh \
  --model experiments/resnet50_semihardmining_20230415_123456/final_model.pth \
  --test-dir data/verification_test \
  --threshold 0.45
```

This is useful when you:
- Want to compare results across different evaluations with a consistent threshold
- Need a specific balance between precision and recall for your application
- Have determined an optimal threshold from previous experiments

#### Evaluation Metrics Interpretation

The evaluation metrics provide insights into model performance:

- **Threshold**: The optimal distance threshold that separates same faces from different faces
- **Accuracy**: Overall accuracy (true positives + true negatives) / total
- **Precision**: How many predicted matches are correct
- **Recall (Sensitivity)**: How many actual matches were correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **ROC Curve**: Shows the trade-off between true positive rate and false positive rate
- **AUC**: Area Under the ROC Curve - higher values indicate better discrimination

## Experiment Tracking

Training results are organized in the experiment directory, with the following structure:

```
experiments/
  └── model_mining-strategy_timestamp/
      ├── checkpoints/
      │   ├── best_model.pth
      │   └── epoch_X.pth
      ├── config.json
      ├── history.json
      ├── training_history.png
      ├── embeddings_tsne.png
      ├── precision_recall_curve.png
      ├── metrics.json
      └── logs/
          └── experiment_timestamp.log
```

The key files are:

- `config.json`: Contains all training parameters
- `history.json`: Training and validation metrics per epoch
- `training_history.png`: Plot of loss curves
- `embeddings_tsne.png`: t-SNE visualization of embeddings
- `precision_recall_curve.png`: Precision-recall curve for face verification
- `metrics.json`: Evaluation metrics on the test set

## Inference

To use the trained model for face recognition:

```bash
python identify_face.py \
  --model experiments/your_experiment_dir/final_model.pth \
  --image path/to/your/image.jpg \
  --data-dir data \
  --threshold 0.5
```

This will:
1. Load the trained model
2. Compare the face in the image with faces in the reference dataset
3. Display the top matches with their confidence scores
4. Save a visualization to `recognition_result.png`

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce the batch size: `--batch-size 8` or even smaller
2. Use a smaller model: `--model mobilenetv2` or `--model resnet18`
3. Reduce image size: `--target-size 160`
4. Use fewer triplets: `--num-triplets 5000`

### Slow Training

To speed up training:

1. Use a pretrained model with frozen backbone: `--model resnet50 --freeze-backbone`
2. Reduce the number of epochs: `--epochs 10`
3. Use a smaller model: `--model mobilenetv2`
4. Use a higher learning rate: `--lr 0.005`

### Poor Performance

If the model is not learning well:

1. Try different mining strategies: `--mining-strategy hard` or `--mining-strategy semi-hard`
2. Adjust the margin: `--margin 0.5` (larger margins can help with harder triplets)
3. Reduce learning rate: `--lr 0.0001`
4. Use more triplets: `--num-triplets 20000`
5. Check data quality: Ensure dataset has multiple images per identity 