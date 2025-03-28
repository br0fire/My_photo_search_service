# Face Recognition System

A face recognition system using siamese neural networks and triplet loss.

## Directory Structure

- `face_recognition.py`: Consolidated model and evaluation functionality
- `evaluate_face_model.sh`: Script for running evaluations
- Training modules: `face_networks.py`, `face_losses.py`, `face_datasets.py`, etc.
- Shell scripts: `train_quick.sh`, `run_recommended_training.sh`, etc.

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training

For quick training:
```bash
./run_quick_training.sh YOUR_ROBOFLOW_API_KEY
```

For recommended training:
```bash
./run_recommended_training.sh YOUR_ROBOFLOW_API_KEY
```

### Evaluation

```bash
./evaluate_face_model.sh --model path/to/model.pth --test-dir path/to/test/dir
```

### Identification

```bash
python identify_face.py --model path/to/model.pth --image path/to/image.jpg
```

For more detailed instructions, see the README and documentation files in the siamese_triplet directory.
