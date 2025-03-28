from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
import time
import numpy as np
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import PIL
import torch
import random
from PIL import  Image
from collections import defaultdict

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_triplets_per_identity=None, random_state=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if random_state:

            random.seed(random_state)
        
        else:
            random.seed(42)

        self.triplets = self.create_triplets(max_triplets_per_identity)  
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        anchor = self.load_image(anchor_path)
        positive = self.load_image(positive_path)
        negative = self.load_image(negative_path)
        
        return anchor, positive, negative
    
    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def create_triplets(self, max_triplets_per_identity):
        # Create a dictionary to map identities to their images
        identity_dict = defaultdict(list)

        # Walk through the directory structure (assumes root_dir contains subdirectories for each identity)
        for root, _, files in os.walk(self.root_dir):
            if files:  # Only process directories that contain images
                identity = os.path.basename(root)
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        identity_dict[identity].append(os.path.join(root, file))

        # Ensure we have at least 2 images per identity for positive pairs
        identity_dict = {k: v for k, v in identity_dict.items() if len(v) >= 2}

        triplets = []
        identities = list(identity_dict.keys())

        for identity in identities:
            # Get all images for this identity
            identity_images = identity_dict[identity]

            # Create positive pairs (anchor and positive from same identity)
            for i in range(len(identity_images)):
                for j in range(i+1, len(identity_images)):  # Limit pairs per anchor
                    anchor = identity_images[i]
                    positive = identity_images[j]

                    # Select a negative from a different identity
                    negative_identity = random.choice([x for x in identities if x != identity])
                    negative = random.choice(identity_dict[negative_identity])

                    triplets.append((anchor, positive, negative))

        # Optional: Balance the dataset by limiting triplets per identity
        if max_triplets_per_identity:
            balanced_triplets = []
            counts = defaultdict(int)
            random.shuffle(triplets)
            for triplet in triplets:
                identity = os.path.basename(os.path.dirname(triplet[0]))
                if counts[identity] < max_triplets_per_identity:
                    balanced_triplets.append(triplet)
                    counts[identity] += 1
            triplets = balanced_triplets

        print(f"Created {len(triplets)} triplets")
        return triplets

def evaluate(model, dataloader, threshold=1.0):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for anchors, positives, negatives in dataloader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            # Get embeddings
            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)
            
            # Calculate distances
            pos_dist = (anchor_emb - positive_emb).pow(2).sum(1)
            neg_dist = (anchor_emb - negative_emb).pow(2).sum(1)
            
            # Count correct predictions
            correct += ((pos_dist < threshold) & (neg_dist >= threshold)).sum().item()
            total += anchors.size(0)
    
    accuracy = correct / total
    return accuracy


def set_seed(seed=42):
    """Set random seed for reproducibility across multiple libraries"""
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    train_dataset = TripletFaceDataset('My-photo-search-2-4/train', max_triplets_per_identity=2000)
    val_dataset = TripletFaceDataset('My-photo-search-2-4/valid', max_triplets_per_identity=500)    
    
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    set_seed(seed=42)

    learning_rate = 0.0001
    weight_decay = 0.001
    margin = 0.2
    num_epochs = 50

    model = InceptionResnetV1(
        classify=False,
        pretrained='vggface2'
    ).to(device)

    #freeze all layers except last linear
    for param in model.parameters():
        param.requires_grad = False

    for param in model.last_linear.parameters():
        param.requires_grad = True

    # Loss and optimizer
    criterion = torch.nn.TripletMarginLoss(margin=margin)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, [40, 50])

    accuracy_list = []
    max_val_accuracy = 0.

    # Training
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        batch_losses = []

        for i, (anchors, positives, negatives) in enumerate(train_dataloader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())

            if i % 100 == 99:  # Print every 100 batches
                avg_loss = running_loss / 100
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], '
                      f'Batch Loss: {avg_loss:.4f}')
                running_loss = 0.0

        epoch_time = time.time() - epoch_start_time
        epoch_loss = sum(batch_losses) / len(batch_losses)
        min_loss = min(batch_losses)
        max_loss = max(batch_losses)

        print('\n' + '='*60)
        print(f'EPOCH {epoch+1} SUMMARY:')
        print(f'Time: {epoch_time:.2f}s | '
              f'Avg Loss: {epoch_loss:.4f} | '
              f'Min Loss: {min_loss:.4f} | '
              f'Max Loss: {max_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        train_accuracy = evaluate(model, train_dataloader)
        print(f'Train Accuracy: {train_accuracy}')

        val_accuracy = evaluate(model, val_dataloader)
        if  val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy 
            torch.save(model.state_dict(), f'/home/melnikum/Projects/Recognition/checkpoints/facenet_checkpoint_last_layer_best.pth')

        print(f'Validation Accuracy: {val_accuracy}')
        print('='*60 + '\n')

        scheduler.step()