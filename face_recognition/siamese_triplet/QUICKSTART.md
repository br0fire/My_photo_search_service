# Face Recognition Quick Start Guide

This quick start guide will help you get up and running with the face recognition system in just a few steps.

## 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd siamese-triplet

# Install dependencies
pip install -r requirements.txt
```

## 2. One-Command Training (Fastest Option)

The fastest way to get started is to use one of our automated training pipelines:

```bash
# For quick results (10 epochs, frozen backbone):
./run_quick_training.sh YOUR_ROBOFLOW_API_KEY

# OR for better results (20 epochs, full fine-tuning):
./run_recommended_training.sh YOUR_ROBOFLOW_API_KEY
```

These scripts will:
1. Download the dataset from Roboflow
2. Train the model with appropriate settings
3. Evaluate the model
4. Create a project archive

## 3. Manual Training Steps

If you prefer to run each step separately:

### 3.1 Download the Dataset

```bash
# Download and prepare the dataset
python download_roboflow_data.py --api-key="YOUR_ROBOFLOW_API_KEY" --output-dir="data"
```

### 3.2 Train a Model (Quick Version)

For quick results, train a model with a pretrained backbone and frozen weights:

```bash
# Train a ResNet-50 model with frozen backbone (fast)
python train_triplet_network.py --model resnet50 --freeze-backbone --epochs 10
```

This will train a model and save the results to `experiments/resnet50_semi-hard_TIMESTAMP/`.

### 3.3 Train a Model (Better Results)

For better results, fine-tune the entire network:

```bash
# Train a ResNet-50 model with full fine-tuning
python train_triplet_network.py --model resnet50 --epochs 20 --mining-strategy semi-hard
```

## 4. Identify Faces

Once training is complete, you can identify faces in new images:

```bash
# Identify faces in a new image
python identify_face.py --model experiments/YOUR_EXPERIMENT_DIR/final_model.pth --image path/to/your/image.jpg
```

## 5. Evaluate Model Performance

Evaluate your trained model on a face verification task:

```bash
# Evaluate the model using the evaluation script
./evaluate_face_model.sh --model experiments/YOUR_EXPERIMENT_DIR/final_model.pth --test-dir path/to/test/data
```

The evaluation script will generate:
- ROC curves showing trade-offs between true and false positive rates
- Precision-recall curves
- Histograms of distance distributions for same/different faces
- Complete metrics in JSON format for further analysis

Results will be saved in the `evaluation_results` directory.

## 6. Experiment with Different Models

Try different model architectures:

```bash
# Train with MobileNetV2 (faster but less accurate)
python train_triplet_network.py --model mobilenetv2

# Train with ResNet-18 (good balance of speed and accuracy)
python train_triplet_network.py --model resnet18

# Train with EfficientNet (efficient and accurate)
python train_triplet_network.py --model efficientnet_b0
```

## 7. Adjust Mining Strategy

Experiment with different triplet mining strategies:

```bash
# Use hard mining (more challenging triplets)
python train_triplet_network.py --model resnet50 --mining-strategy hard

# Use semi-hard mining (recommended)
python train_triplet_network.py --model resnet50 --mining-strategy semi-hard

# Use random triplets (baseline)
python train_triplet_network.py --model resnet50 --mining-strategy all
```

## 8. Create a ZIP of the Project

To share the project:

```bash
python create_zip.py --output-path face_recognition_project.zip
```

## Next Steps

For more detailed instructions, see:
- [Training Guide](TRAINING.md) for advanced training options
- [README.md](README.md) for project overview

For help with the project, refer to the troubleshooting section in the [Training Guide](TRAINING.md#troubleshooting). 