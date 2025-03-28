#!/bin/bash

# Train Siamese Network with Triplet Loss (Quick Version)
#
# This script provides parameters for fast training with a frozen backbone,
# which is ideal for quick experimentation or limited computational resources.
#
# Usage:
#   ./train_quick.sh

# Navigate to project directory (if not already there)
cd "$(dirname "$0")"

# Create a timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting quick training at $TIMESTAMP"

# Run the training script with quick training parameters
python train_triplet_network.py \
  --data-dir data \
  --model resnet50 \
  --embedding-dim 128 \
  --mining-strategy semi-hard \
  --batch-size 64 \
  --epochs 10 \
  --lr 0.005 \
  --margin 0.3 \
  --scheduler cosine \
  --freeze-backbone \
  --output-dir experiments \
  --exp-name "resnet50_quick_$TIMESTAMP"

echo "Quick training completed. Results saved to experiments/resnet50_quick_$TIMESTAMP" 