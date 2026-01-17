import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Model Definitions
# =========================

class MobileNetV2WithDropout(nn.Module):
    def __init__(self, base_model, dropout_p=0.4):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # Forward through features
        x = self.base.features(x)
        
        # Global average pooling (3D)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        
        # Apply additional dropout before classifier
        x = self.dropout(x)
        
        # Final classifier (already has dropout inside)
        x = self.base.classifier(x)
        return x


# =========================
# Dataset (Modified for 3D Conv)
# =========================

class NiADVideoDataset(Dataset):
    def __init__(self, root_dir, phase="Testing", frames_per_clip=16):
        self.root_dir = root_dir
        self.phase = phase
        self.frames_per_clip = frames_per_clip
        self.classes = ["Normal", "Accident"]
        self.samples = []

        phase_dir = os.path.join(root_dir, phase)
        print(f"\nLoading {phase} dataset from: {phase_dir}")

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(phase_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Directory not found: {class_dir}")
                continue

            files = [
                f for f in os.listdir(class_dir)
                if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]
            for fname in files:
                self.samples.append((os.path.join(class_dir, fname), label))

            print(f"  {class_name}: {len(files)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        
        # Convert to tensor: (frames, height, width, channels) -> (channels, frames, height, width)
        frames = torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
        frames = frames / 255.0
        return frames, label, video_path

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while len(frames) < self.frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        while len(frames) < self.frames_per_clip:
            frames.append(np.zeros((112, 112, 3), dtype=np.uint8))

        return frames


# =========================
# Model Loading
# =========================

def load_model(checkpoint_path, device):
    """Load the trained MobileNetV2 model from checkpoint"""
    print(f"\nLoading model from: {checkpoint_path}")
    
    import sys
    sys.path.append('/home/23ucc504/Accident')
    from models.mobilenetv2 import get_model
    
    # Build base model
    base_model = get_model(
        num_classes=2,
        sample_size=112,
        width_mult=1.0
    )
    
    # Wrap with dropout
    model = MobileNetV2WithDropout(base_model, dropout_p=0.4).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return model


# =========================
# Visualization Functions
# =========================

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title("Confusion Matrix - MobileNetV2")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path="roc_curve.png"):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - MobileNetV2")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"ROC curve saved to {save_path}")
    plt.close()


# =========================
# Testing Function
# =========================

def test_model(
    model,
    test_loader,
    device,
    save_results=True,
    output_dir="test_results"
):
    """Test the model and generate comprehensive metrics"""
    
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []
    
    class_correct = [0, 0]
    class_total = [0, 0]
    
    print("\n" + "="*50)
    print("Starting Testing...")
    print("="*50)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, paths) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of accident class
            all_paths.extend(paths)
            
            # Per-class accuracy
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # AUC-ROC
    auc = roc_auc_score(all_labels, all_probs)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS - MobileNetV2")
    print("="*50)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC-ROC Score: {auc:.4f}")
    print(f"\nWeighted Precision: {overall_precision:.4f}")
    print(f"Weighted Recall: {overall_recall:.4f}")
    print(f"Weighted F1-Score: {overall_f1:.4f}")
    
    print("\n" + "-"*50)
    print("Per-Class Metrics:")
    print("-"*50)
    classes = ["Normal", "Accident"]
    for i, class_name in enumerate(classes):
        print(f"\n{class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-Score: {f1[i]:.4f}")
        print(f"  Support: {support[i]}")
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  Accuracy: {acc:.2f}%")
    
    print("\n" + "-"*50)
    print("Confusion Matrix:")
    print("-"*50)
    print(f"                Predicted")
    print(f"              Normal  Accident")
    print(f"Actual Normal   {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"      Accident  {cm[1][0]:5d}    {cm[1][1]:5d}")
    
    # Save results
    if save_results:
        # Plot confusion matrix
        plot_confusion_matrix(cm, classes, os.path.join(output_dir, "confusion_matrix.png"))
        
        # Plot ROC curve
        plot_roc_curve(all_labels, all_probs, os.path.join(output_dir, "roc_curve.png"))
        
        # Save detailed results to text file
        results_file = os.path.join(output_dir, "test_results.txt")
        with open(results_file, "w") as f:
            f.write("="*50 + "\n")
            f.write("TEST RESULTS - MobileNetV2\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"AUC-ROC Score: {auc:.4f}\n\n")
            f.write(f"Weighted Precision: {overall_precision:.4f}\n")
            f.write(f"Weighted Recall: {overall_recall:.4f}\n")
            f.write(f"Weighted F1-Score: {overall_f1:.4f}\n\n")
            f.write("-"*50 + "\n")
            f.write("Per-Class Metrics:\n")
            f.write("-"*50 + "\n")
            for i, class_name in enumerate(classes):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {precision[i]:.4f}\n")
                f.write(f"  Recall: {recall[i]:.4f}\n")
                f.write(f"  F1-Score: {f1[i]:.4f}\n")
                f.write(f"  Support: {support[i]}\n")
            f.write("\n" + "-"*50 + "\n")
            f.write("Confusion Matrix:\n")
            f.write("-"*50 + "\n")
            f.write(f"                Predicted\n")
            f.write(f"              Normal  Accident\n")
            f.write(f"Actual Normal   {cm[0][0]:5d}    {cm[0][1]:5d}\n")
            f.write(f"      Accident  {cm[1][0]:5d}    {cm[1][1]:5d}\n")
        
        print(f"\nResults saved to {results_file}")
        
        # Save predictions to CSV
        predictions_file = os.path.join(output_dir, "predictions.csv")
        with open(predictions_file, "w") as f:
            f.write("Video Path,True Label,Predicted Label,Confidence,Correct\n")
            for path, true_label, pred_label, prob in zip(all_paths, all_labels, all_preds, all_probs):
                confidence = prob if pred_label == 1 else (1 - prob)
                correct = "Yes" if true_label == pred_label else "No"
                f.write(f"{path},{classes[true_label]},{classes[pred_label]},{confidence:.4f},{correct}\n")
        
        print(f"Predictions saved to {predictions_file}")
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs
    }


# =========================
# Main Testing Script
# =========================

def main():
    # Configuration
    ROOT_DIR = "/home/23ucc504/Accident/accident_prediction/NiAD_Large_Videos"
    CHECKPOINT_PATH = "best_mobilenet.pth"
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda:5")
    NUM_WORKERS = 2
    OUTPUT_DIR = "test_results_mobilenet"
    FRAMES_PER_CLIP = 16  # Must match training
    
    # Load test dataset
    test_dataset = NiADVideoDataset(
        root_dir=ROOT_DIR,
        phase="Testing",
        frames_per_clip=FRAMES_PER_CLIP
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    print(f"\nTotal test samples: {len(test_dataset)}")
    
    # Load model
    model = load_model(CHECKPOINT_PATH, DEVICE)
    
    # Run testing
    results = test_model(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        save_results=True,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*50)
    print("Testing Complete!")
    print("="*50)


if __name__ == "__main__":
    main()