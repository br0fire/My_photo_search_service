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
from argparse import ArgumentParser

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

def evaluate_image(model, face_image_path, dataloader, threshold=1.0):
      
    model.eval()
    TP, FP, FN, TN = 0, 0, 0, 0
    transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    face_image = PIL.Image.open('/home/melnikum/Projects/Recognition/yura_face_image.jpg')
    face = transf(face_image).to(device)
    feature_face = model(face[None])
    
    with torch.no_grad():
        for images, target in dataloader:
            
            images, target = images.to(device), target.to(device)
            encoded_images = model(images)

            distances = torch.cdist(feature_face[None], encoded_images)

            TP += ((distances < threshold) & (target == 1)).sum()
            FP += ((distances < threshold) & (target == 0)).sum()
            FN += ((distances >= threshold) & (target == 1)).sum()
            TN += ((distances >= threshold) & (target == 0)).sum()

        
    print(f'Precision : {TP / (TP + FP):.3f}')
    print(f'Recal : {TP / (TP + FN):.3f}')
    print(f'Accuracy : {(TP + TN) / (TP + FN + FP + TN):.3f}')


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--database_path", type=str)

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    loaded_model = InceptionResnetV1(classify=False, pretrained='vggface2').to(device)
    loaded_model.load_state_dict(torch.load(args.checkpoint_path))
    loaded_model.eval()   
    
    data_dir = args.database_path
    test_dataset = datasets.ImageFolder(data_dir, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
    evaluate_image(loaded_model, args.image_path, test_dataloader)