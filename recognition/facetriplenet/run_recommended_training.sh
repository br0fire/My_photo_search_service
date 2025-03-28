#!/bin/bash

# Recommended Face Recognition Training Pipeline
#
# This script runs the entire face recognition pipeline with recommended settings:
# 1. Downloads data from Roboflow
# 2. Trains a face recognition model with full fine-tuning (better results)
#
# Usage:
#   ./run_recommended_training.sh YOUR_ROBOFLOW_API_KEY

# Check if API key is provided
if [ -z "$1" ]; then
  echo "Error: Roboflow API key is required."
  echo "Usage: ./run_recommended_training.sh YOUR_ROBOFLOW_API_KEY"
  exit 1
fi

API_KEY="$1"
DATA_DIR="data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="recommended_training_${TIMESTAMP}.log"

echo "Starting recommended face recognition training pipeline at $(date)" | tee -a "$LOG_FILE"
echo "====================================================" | tee -a "$LOG_FILE"

# Step 1: Download and prepare dataset
echo "Step 1: Downloading dataset from Roboflow..." | tee -a "$LOG_FILE"
python download_roboflow_data.py --api-key="$API_KEY" --output-dir="$DATA_DIR" | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "Error: Failed to download dataset. Check the log file for details." | tee -a "$LOG_FILE"
  exit 1
fi

echo "Dataset download completed successfully." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 2: Train the model with recommended settings (full fine-tuning)
echo "Step 2: Training the face recognition model (recommended version)..." | tee -a "$LOG_FILE"
./train_recommended.sh | tee -a "$LOG_FILE"
MODEL_DIR="resnet50_recommended_${TIMESTAMP}"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "Error: Training failed. Check the log file for details." | tee -a "$LOG_FILE"
  exit 1
fi

echo "Model training completed successfully." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "====================================================" | tee -a "$LOG_FILE"
echo "Recommended face recognition training pipeline completed successfully at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to experiments/$MODEL_DIR" | tee -a "$LOG_FILE"
echo "Full log saved to $LOG_FILE" | tee -a "$LOG_FILE"

# Display information about using the trained model
echo ""
echo "To identify faces using your trained model, run:"
echo "python identify_face.py --model experiments/$MODEL_DIR/final_model.pth --image path/to/your/image.jpg" 