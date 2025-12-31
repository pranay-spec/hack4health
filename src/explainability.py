import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .config import DEVICE, IMG_SIZE

# D. Grad-CAM Heatmaps (One per Class)
def get_gradcam(model, image_tensor, label):
    target_layer = model.features[-1]
    gradients = []; activations = []
    def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
    def forward_hook(module, input, output): activations.append(output)
    handle_b = target_layer.register_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)
    
    model.zero_grad()
    output = model(image_tensor.unsqueeze(0))
    score = output[0, label]
    score.backward()
    
    grads = gradients[0][0]; acts = activations[0][0]
    weights = torch.mean(grads, dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], device=DEVICE)
    for i, w in enumerate(weights): cam += w * acts[i, :, :]
    cam = torch.relu(cam); cam = cam - torch.min(cam); cam = cam / torch.max(cam)
    handle_b.remove(); handle_f.remove()
    return cam.cpu().detach().numpy()

def visualize_gradcam(model, vis_loader, classes):
    print("\n🧠 Generating Heatmaps for Each Class...")
    # Find one example for each class
    found_classes = set()
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    row = 0

    for images, labels in vis_loader:
        label_val = labels.item()
        if label_val not in found_classes and row < 4:
            found_classes.add(label_val)
            images = images.to(DEVICE)
            heatmap = get_gradcam(model, images[0], label_val)
            
            img_disp = images[0].permute(1, 2, 0).cpu().numpy()
            img_disp = (img_disp * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_disp = np.clip(img_disp, 0, 1)
            hm_res = np.array(Image.fromarray(np.uint8(255*heatmap)).resize((IMG_SIZE,IMG_SIZE), Image.BILINEAR))/255.0
            
            axes[row, 0].imshow(img_disp); axes[row, 0].set_title(f"Class: {classes[label_val]}")
            axes[row, 0].axis('off')
            axes[row, 1].imshow(img_disp); axes[row, 1].imshow(hm_res, cmap='jet', alpha=0.5)
            axes[row, 1].set_title("Grad-CAM Focus"); axes[row, 1].axis('off')
            row += 1
        if len(found_classes) == 4: break
    plt.tight_layout()
    plt.savefig('gradcam_explained.png')
    plt.show()
