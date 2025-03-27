#!/bin/bash

# Train Siamese Network with Triplet Loss (Recommended Parameters)
#
# This script provides the recommended training parameters for the best results
# with a good balance between training time and accuracy.
#
# Usage:
#   ./train_recommended.sh

# Navigate to project directory (if not already there)
cd "$(dirname "$0")"

# Create a timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting training at $TIMESTAMP"

# Run the training script with recommended parameters
python train_triplet_network.py \
  --data-dir data \
  --model resnet50 \
  --embedding-dim 256 \
  --mining-strategy semi-hard \
  --batch-size 32 \
  --epochs 20 \
  --lr 0.001 \
  --margin 0.3 \
  --scheduler cosine \
  --output-dir experiments \
  --exp-name "resnet50_recommended_$TIMESTAMP"

echo "Training completed. Results saved to experiments/resnet50_recommended_$TIMESTAMP" 