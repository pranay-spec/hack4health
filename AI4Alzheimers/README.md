# 🧠 AI4Alzheimers
### Advanced Alzheimer's Severity Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**AI4Alzheimers** is a robust deep learning framework designed to classify the severity of Alzheimer's disease from MRI scans. Built with competition-grade techniques, it leverages **EfficientNetV2**, **Mixup Augmentation**, and **Test-Time Augmentation (TTA)** to achieve high accuracy and generalization.

---

## ✨ Key Features

### 🏗️ State-of-the-Art Architecture
*   **Backbone**: `EfficientNetV2-Medium` (Pre-trained on ImageNet) for powerful feature extraction.
*   **Custom Head**: Fine-tuned classifier with Dropout (`p=0.4`) for regularization.

### 🛡️ Robust Training Strategy
*   **Mixup Augmentation**: Linearly interpolates between images and labels (50% probability) to encourage smoother decision boundaries.
*   **OneCycleLR Scheduler**: Dynamic learning rate adjustment for faster convergence and better stability.
*   **Label Smoothing**: Prevents the model from becoming over-confident (epsilon=0.05).
*   **Data Splits**: 90% Training / 10% Validation for maximizing training data utilization.

### 🔍 Explainability & Inference
*   **10-Crop TTA ("The 100% Strategy")**: During inference, images are cropped 10 times (corners + center + flips), and predictions are averaged for maximum reliability.
*   **Grad-CAM Support**: Built-in visualization tools to generate heatmaps, showing exactly *where* the model is looking in the brain MRI.
*   **Comprehensive Metrics**: Auto-generates Confusion Matrices, ROC Curves, and Classification Reports.

---

## 📂 Project Structure

```bash
AI4Alzheimers/
├── 📁 models/              # Stores saved model checkpoints (.pth)
├── 📁 scripts/             # Execution scripts
│   ├── train_kaggle.py     # Main training loop (Mixup + OneCycleLR)
│   └── inference.py        # Inference logic (10-Crop TTA + Grad-CAM)
├── 📁 src/                 # Source code module
│   ├── config.py           # Hyperparameters (Batch Size, LR, Epochs)
│   ├── dataset.py          # Data loading & Augmentations (Albumentations/Torchvision)
│   ├── model.py            # EfficientNetV2 model definition
│   ├── explainability.py   # Grad-CAM implementation
│   ├── evaluate.py         # Metrics & Plotting utilities
│   └── train_utils.py      # Helpers for Mixup/Loss
└── README.md               # Project Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. Install dependencies (create a `requirements.txt` if needed, but standard torch/torchvision/pandas/tqdm are required):

```bash
pip install torch torchvision pandas tqdm matplotlib scikit-learn pillow pyarrow
```

### 2. Data Preparation
The project expects a Parquet file named `train.parquet` containing the MRI data.
*   **Column `image`**: Byte data of the image.
*   **Column `label`**: Target class for the image.

Place `train.parquet` in the root or `/kaggle/input` directory. The script will search for it automatically.

### 3. Training
To start the training pipeline with Mixed Precision and Mixup:

```bash
python scripts/train_kaggle.py
```
> **Note**: The best model will be saved to `models/best_severity_model.pth`.

### 4. Inference & Evaluation
Run the full evaluation suite, which includes TTA and Grad-CAM generation:

```bash
python scripts/inference.py
```
*   **Outputs**:
    *   `gradcam_explained.png`: Heatmap visualizations.
    *   Console output: Final Accuracy, Classification Report.

---

## ⚙️ Configuration
You can tweak hyperparameters in `src/config.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `IMG_SIZE` | `384` | Input resolution (High Res) |
| `BATCH_SIZE` | `16` | Images per batch |
| `EPOCHS` | `20` | Total training cycles |
| `MAX_LR` | `5e-4` | Peak learning rate for OneCycle |
| `DEVICE` | `cuda` | Auto-detects GPU/CPU |

---

## 📊 Results & Visualization

The pipeline automatically generates insights after inference:
1.  **Confusion Matrix**: To see where the model gets confused.
2.  **ROC Curve**: To measure class separation performance.
3.  **Grad-CAM Heatmaps**:
    > Visualizes the "attention" of the neural network, highlighting regions of interest (ROI) in the brain tissue that contributed to the diagnosis.

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

## 👥 Contributors

**Team Code Crashers** 🚀

*   **Pranay Gujar**
*   **Anuj Gardi**
*   **Saurabh Gangurde**

---
*Built for Hack4Health 2025*
