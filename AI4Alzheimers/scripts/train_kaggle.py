import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import numpy as np

# Adjust path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import DEVICE, EPOCHS, MAX_LR
from src.dataset import get_dataloaders
from src.model import build_model
from src.train_utils import mixup_data, mixup_criterion

def train():
    print(f"🚀 Starting Ultimate Run (V2-Medium + 10-Crop TTA) on: {DEVICE}")

    # ==========================================
    # 2. DATA LOAD & PREPROCESSING
    # ==========================================
    mri_path = None
    # Flexible path searching
    search_paths = ['/kaggle/input', '.', '..', 'C:/Users/Admin/.gemini/antigravity/scratch/hack4health']
    for search_root in search_paths:
        if not os.path.exists(search_root): continue
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if file.endswith("train.parquet"):
                    mri_path = os.path.join(root, file)
                    break
            if mri_path: break
        if mri_path: break
            
    if not mri_path: 
        print("❌ train.parquet not found in common locations!")
        return

    print(f"📂 Loading data from: {mri_path}")
    df = pd.read_parquet(mri_path)
    
    train_loader, val_loader, full_dataset, _ = get_dataloaders(df)

    # ==========================================
    # 3. MODEL
    # ==========================================
    NUM_CLASSES = len(full_dataset.classes)
    model = build_model(NUM_CLASSES)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) 
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    # ==========================================
    # 4. TRAINING LOOP
    # ==========================================
    best_acc = 0.0
    print(f"🔥 Training for {EPOCHS} Epochs...")

    for epoch in range(EPOCHS):
        model.train()
        correct = 0; total = 0
        loop = tqdm(train_loader, leave=False)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Mixup 50% of the time
            if np.random.rand() > 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track Acc
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")

        # Validation
        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_severity_model.pth')
            torch.save(model.state_dict(), save_path)
            
        val_color = "🟢" if val_acc > 99.0 else "🔴"
        print(f"Epoch {epoch+1} | Train: {100*correct/total:.2f}% | {val_color} Val: {val_acc:.2f}% (Best: {best_acc:.2f}%)")

    print(f"🏆 Best Standard Accuracy: {best_acc:.2f}%")
    print(f"💾 Model saved to models/best_severity_model.pth")

if __name__ == "__main__":
    train()
