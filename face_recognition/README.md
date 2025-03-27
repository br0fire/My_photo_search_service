# Face Recognition with Siamese Networks

Face recognition system using siamese neural networks and triplet loss.

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Evaluation

Run the evaluation script:
```bash
./evaluate_face_model.sh --model path/to/model.pth --test-dir path/to/test/dir
```

Optional parameters:
- `--output-dir`: Directory to save results (default: evaluation_results)
- `--threshold`: Custom threshold value (default: automatically determined)

## File Overview

- `face_recognition.py`: Consolidated script with model architecture and evaluation functionality
- `evaluate_face_model.sh`: Convenient shell script for running evaluations

## Threshold Values

When you run the evaluation, the script will either:
1. Automatically find the optimal threshold (default)
2. Use your custom threshold value (if provided)

The threshold determines whether two faces match:
- If the distance between face embeddings is < threshold: SAME person
- If the distance between face embeddings is ≥ threshold: DIFFERENT person

## Output

The evaluation produces:
1. Metrics JSON file with:
   - Threshold value
   - Accuracy, precision, recall, F1 score
   - Confusion matrix values (TP, FP, TN, FN)

2. Visualization plots:
   - ROC curve
   - Precision-recall curve
   - Distance distributions for same/different faces 