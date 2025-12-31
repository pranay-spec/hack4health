import os
import io
import copy
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from .config import IMG_SIZE, BATCH_SIZE

class AlzheimerMRIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(dataframe['label'].unique())
        self.label_map = {label: i for i, label in enumerate(self.classes)}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row['image']['bytes'] if isinstance(row['image'], dict) else row['image']
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.label_map[row['label']]

def get_transforms():
    # Training Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation Transforms (Standard)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 10-CROP Transform (The 100% Strategy)
    ten_crop_transform = transforms.Compose([
        transforms.Resize((420, 420)), # Resize larger first
        transforms.TenCrop(IMG_SIZE),  # Crop to 384
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(crop)) for crop in crops
        ]))
    ])
    
    return train_transform, val_transform, ten_crop_transform

def get_dataloaders(df):
    train_transform, val_transform, _ = get_transforms()
    
    full_dataset = AlzheimerMRIDataset(df, transform=train_transform)
    train_size = int(0.90 * len(full_dataset)) # 90% Training for Max Accuracy
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Decouple Validation Dataset
    val_subset.dataset = copy.copy(full_dataset)
    val_subset.dataset.transform = val_transform 

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, full_dataset, val_subset
