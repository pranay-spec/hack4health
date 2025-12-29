import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# Adjust path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import DEVICE
from src.dataset import get_dataloaders
from src.model import build_model
from src.evaluate import plot_confusion_matrix, print_classification_report, plot_roc_curve
from src.explainability import visualize_gradcam

def inference():
    print("\n🚀 Running 10-CROP TTA (The 100% Strategy)...")
    
    # 1. Load Data (Simplified for inference env assuming parquet exists)
    mri_path = None
    search_paths = ['/kaggle/input', '.', '..', 'C:/Users/Admin/.gemini/antigravity/scratch/hack4health']
    for search_root in search_paths:
        if not os.path.exists(search_root): continue
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if file.endswith("train.parquet"):
                    mri_path = os.path.join(root, file)
                    break
            if mri_path: break

    if not mri_path:
        print("❌ train.parquet not found!")
        return

    df = pd.read_parquet(mri_path)
    # We need the full dataset to get classes and validation config
    _, _, full_dataset, val_subset = get_dataloaders(df)
    
    # 2. Load Model
    NUM_CLASSES = len(full_dataset.classes)
    model = build_model(NUM_CLASSES)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_severity_model.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. Setup 10-Crop Loader
    # Note: get_dataloaders returns standard val loader, we need to manually set 10-crop transform here
    from src.dataset import get_transforms
    _, _, ten_crop_transform = get_transforms()
    
    val_subset.dataset.transform = ten_crop_transform
    loader_10crop = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=2)

    # 4. TTA Loop
    tta_correct = 0; tta_total = 0
    all_preds = []; all_labels = []; all_probs = []

    with torch.no_grad():
        loop = tqdm(loader_10crop)
        for inputs, labels in loop:
            # Inputs shape: [Batch, 10, 3, 384, 384]
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w).to(DEVICE) # Fuse batch and crops
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            outputs = outputs.view(bs, ncrops, -1) # Unfuse
            outputs = outputs.mean(1) # Average the 10 crops
            
            _, predicted = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            
            tta_total += labels.size(0)
            tta_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            loop.set_description(f"TTA Acc: {100*tta_correct/tta_total:.2f}%")

    final_acc = 100 * tta_correct / tta_total
    print(f"\n🌟 FINAL ULTIMATE ACCURACY: {final_acc:.2f}%")

    # 5. Full Analysis
    print("\n📊 Generating All Analysis Charts...")
    plot_confusion_matrix(all_labels, all_preds, full_dataset.classes, final_acc)
    print_classification_report(all_labels, all_preds, full_dataset.classes)
    plot_roc_curve(all_labels, all_probs, full_dataset.classes)
    
    # 6. GradCAM
    # Need standard transform for visualization
    from src.dataset import get_transforms
    _, val_transform, _ = get_transforms()
    val_subset.dataset.transform = val_transform
    vis_loader = DataLoader(val_subset, batch_size=1, shuffle=True)
    visualize_gradcam(model, vis_loader, full_dataset.classes)

if __name__ == "__main__":
    inference()
