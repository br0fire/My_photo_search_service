#!/bin/bash

# Default values
MODEL_PATH=""
TEST_DIR=""
OUTPUT_DIR="evaluation_results"
THRESHOLD=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --test-dir)
      TEST_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model      Path to trained model (required)"
      echo "  --test-dir   Directory with test data (required)"
      echo "  --output-dir Output directory for evaluation results (default: evaluation_results)"
      echo "  --threshold  Custom threshold value for face verification (optional, default: auto-determined)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
  echo "Error: Model path is required"
  echo "Use --help for usage information"
  exit 1
fi

if [ -z "$TEST_DIR" ]; then
  echo "Error: Test directory is required"
  echo "Use --help for usage information"
  exit 1
fi

# Create log directory
mkdir -p "logs"
LOG_FILE="logs/evaluation_$(date +%Y%m%d_%H%M%S).log"

echo "Starting face recognition model evaluation..."
echo "Model: $MODEL_PATH"
echo "Test directory: $TEST_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$THRESHOLD" ]; then
  echo "Using custom threshold: $THRESHOLD"
fi
echo "Log file: $LOG_FILE"

# Prepare evaluation command
EVAL_CMD="python face_recognition.py --model \"$MODEL_PATH\" --test-dir \"$TEST_DIR\" --output-dir \"$OUTPUT_DIR\""

# Add threshold argument if provided
if [ -n "$THRESHOLD" ]; then
  EVAL_CMD="$EVAL_CMD --threshold $THRESHOLD"
fi

# Run evaluation
eval $EVAL_CMD 2>&1 | tee "$LOG_FILE"

# Check if evaluation was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "Evaluation completed successfully!"
  echo "Results are available in: $OUTPUT_DIR"
  echo "Metrics: $OUTPUT_DIR/evaluation_metrics.json"
  echo "Plots: $OUTPUT_DIR/evaluation_plots.png"
else
  echo "Evaluation failed. Check log file: $LOG_FILE"
  exit 1
fi 