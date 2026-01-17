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
# Model Definitions (C3D)
# =========================

class C3DWithDropout(nn.Module):
    def __init__(self, base_model, dropout_p=0.5):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # C3D forward pass through conv layers
        x = self.base.conv1(x)
        x = self.base.pool1(x)
        
        x = self.base.conv2(x)
        x = self.base.pool2(x)
        
        x = self.base.conv3a(x)
        x = self.base.conv3b(x)
        x = self.base.pool3(x)
        
        x = self.base.conv4a(x)
        x = self.base.conv4b(x)
        x = self.base.pool4(x)
        
        x = self.base.conv5a(x)
        x = self.base.conv5b(x)
        x = self.base.pool5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers with dropout
        x = F.relu(self.base.fc6(x))
        x = self.dropout(x)
        
        x = F.relu(self.base.fc7(x))
        x = self.dropout(x)
        
        x = self.base.fc8(x)
        return x


# =========================
# Dataset (SAME AS TRAINING - CRITICAL!)
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
        
        # Convert to tensor and normalize (SAME AS TRAINING - CRITICAL!)
        frames = torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
        
        # Normalize using ImageNet stats (MUST MATCH TRAINING)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames = (frames / 255.0 - mean) / std
        
        return frames, label, video_path

    def _load_video(self, path):
        """Load video with uniform frame sampling (SAME AS TRAINING)"""
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        # Sample frames uniformly across video (SAME AS TRAINING)
        if total_frames >= self.frames_per_clip:
            # Sample evenly distributed frames
            indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (112, 112))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        else:
            # If video has fewer frames, read all and then pad
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        # Pad if necessary
        while len(frames) < self.frames_per_clip:
            frames.append(np.zeros((112, 112, 3), dtype=np.uint8))

        return frames


# =========================
# Model Loading (Updated for Transfer Learning)
# =========================

def load_model(checkpoint_path, device):
    """Load the trained C3D model from checkpoint"""
    print(f"\nLoading model from: {checkpoint_path}")
    
    import models.c3d as c3d_mod
    
    # Build base C3D model
    if hasattr(c3d_mod, 'C3D'):
        try:
            base_model = c3d_mod.C3D(num_classes=2)
        except TypeError:
            base_model = c3d_mod.C3D()
    elif hasattr(c3d_mod, 'get_model'):
        try:
            base_model = c3d_mod.get_model(num_classes=2)
        except TypeError:
            base_model = c3d_mod.get_model()
    elif hasattr(c3d_mod, 'generate_model'):
        try:
            base_model = c3d_mod.generate_model(num_classes=2)
        except TypeError:
            base_model = c3d_mod.generate_model()
    else:
        raise RuntimeError("Cannot find C3D model in models.c3d module")
    
    # Check and modify final layer if needed
    if hasattr(base_model, 'fc') and base_model.fc.out_features != 2:
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, 2)
        print(f"Modified final FC layer: {in_features} -> 2 classes")
    elif hasattr(base_model, 'fc8') and base_model.fc8.out_features != 2:
        in_features = base_model.fc8.in_features
        base_model.fc8 = nn.Linear(in_features, 2)
        print(f"Modified final FC layer (fc8): {in_features} -> 2 classes")
    
    # Wrap with dropout (SAME AS TRAINING)
    model = C3DWithDropout(base_model, dropout_p=0.5).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"? Model loaded from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print(f"? Model weights loaded")
    
    return model


# =========================
# Visualization Functions
# =========================

def plot_confusion_matrix(cm, classes, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={"size": 16}
    )
    plt.title("Confusion Matrix", fontsize=18, fontweight='bold')
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"? Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path="roc_curve.png"):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=3, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"? ROC curve saved to {save_path}")
    plt.close()


def plot_per_class_metrics(precision, recall, f1, classes, save_path="per_class_metrics.png"):
    """Plot per-class precision, recall, and F1-score"""
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Classes', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"? Per-class metrics saved to {save_path}")
    plt.close()


# =========================
# Testing Function
# =========================

def test_model(
    model,
    test_loader,
    device,
    save_results=True,
    output_dir="test_results_c3d"
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
    
    print("\n" + "="*60)
    print("STARTING TESTING")
    print("="*60)
    
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
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches ({100*(batch_idx+1)/len(test_loader):.1f}%)")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
        print("Warning: Could not calculate AUC-ROC (insufficient data)")
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 60)
    print(f"{'Overall Accuracy':<30} {accuracy * 100:>9.2f}%")
    print(f"{'AUC-ROC Score':<30} {auc:>10.4f}")
    print(f"{'Weighted Precision':<30} {overall_precision:>10.4f}")
    print(f"{'Weighted Recall':<30} {overall_recall:>10.4f}")
    print(f"{'Weighted F1-Score':<30} {overall_f1:>10.4f}")
    
    print("\n" + "="*60)
    print("PER-CLASS METRICS")
    print("="*60)
    classes = ["Normal", "Accident"]
    
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Accuracy':>10}")
    print("-" * 75)
    for i, class_name in enumerate(classes):
        class_acc = 100.0 * class_correct[i] / max(1, class_total[i])
        print(f"{class_name:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d} {class_acc:>9.2f}%")
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"\n{'':>15} Predicted")
    print(f"{'':>15} {'Normal':>10} {'Accident':>10}")
    print(f"{'Actual':>10}")
    print(f"  {'Normal':>13} {cm[0][0]:>10d} {cm[0][1]:>10d}")
    print(f"  {'Accident':>13} {cm[1][0]:>10d} {cm[1][1]:>10d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n" + "="*60)
    print("ADDITIONAL METRICS")
    print("="*60)
    print(f"{'True Negatives (TN)':<30} {tn:>10d}")
    print(f"{'False Positives (FP)':<30} {fp:>10d}")
    print(f"{'False Negatives (FN)':<30} {fn:>10d}")
    print(f"{'True Positives (TP)':<30} {tp:>10d}")
    print(f"{'Specificity':<30} {specificity:>10.4f}")
    print(f"{'Sensitivity/Recall':<30} {sensitivity:>10.4f}")
    
    # Save results
    if save_results:
        # Plot confusion matrix
        plot_confusion_matrix(cm, classes, os.path.join(output_dir, "confusion_matrix.png"))
        
        # Plot ROC curve
        if auc > 0:
            plot_roc_curve(all_labels, all_probs, os.path.join(output_dir, "roc_curve.png"))
        
        # Plot per-class metrics
        plot_per_class_metrics(precision, recall, f1, classes, 
                              os.path.join(output_dir, "per_class_metrics.png"))
        
        # Save detailed results to text file
        results_file = os.path.join(output_dir, "test_results.txt")
        with open(results_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("TEST RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"AUC-ROC Score: {auc:.4f}\n\n")
            f.write(f"Weighted Precision: {overall_precision:.4f}\n")
            f.write(f"Weighted Recall: {overall_recall:.4f}\n")
            f.write(f"Weighted F1-Score: {overall_f1:.4f}\n\n")
            
            f.write("="*60 + "\n")
            f.write("Per-Class Metrics:\n")
            f.write("="*60 + "\n")
            for i, class_name in enumerate(classes):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {precision[i]:.4f}\n")
                f.write(f"  Recall: {recall[i]:.4f}\n")
                f.write(f"  F1-Score: {f1[i]:.4f}\n")
                f.write(f"  Support: {support[i]}\n")
                if class_total[i] > 0:
                    class_acc = 100.0 * class_correct[i] / class_total[i]
                    f.write(f"  Accuracy: {class_acc:.2f}%\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("Confusion Matrix:\n")
            f.write("="*60 + "\n")
            f.write(f"                Predicted\n")
            f.write(f"              Normal  Accident\n")
            f.write(f"Actual Normal   {cm[0][0]:5d}    {cm[0][1]:5d}\n")
            f.write(f"      Accident  {cm[1][0]:5d}    {cm[1][1]:5d}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("Additional Metrics:\n")
            f.write("="*60 + "\n")
            f.write(f"True Negatives (TN): {tn}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"False Negatives (FN): {fn}\n")
            f.write(f"True Positives (TP): {tp}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write(f"Sensitivity: {sensitivity:.4f}\n")
        
        print(f"\n? Results saved to {results_file}")
        
        # Save predictions to CSV
        predictions_file = os.path.join(output_dir, "predictions.csv")
        with open(predictions_file, "w") as f:
            f.write("Video Path,True Label,Predicted Label,Accident Probability,Confidence,Correct\n")
            for path, true_label, pred_label, prob in zip(all_paths, all_labels, all_preds, all_probs):
                confidence = prob if pred_label == 1 else (1 - prob)
                correct = "Yes" if true_label == pred_label else "No"
                f.write(f'"{path}",{classes[true_label]},{classes[pred_label]},{prob:.4f},{confidence:.4f},{correct}\n')
        
        print(f"? Predictions saved to {predictions_file}")
        
        # Save misclassified samples
        misclassified_file = os.path.join(output_dir, "misclassified.csv")
        with open(misclassified_file, "w") as f:
            f.write("Video Path,True Label,Predicted Label,Confidence\n")
            for path, true_label, pred_label, prob in zip(all_paths, all_labels, all_preds, all_probs):
                if true_label != pred_label:
                    confidence = prob if pred_label == 1 else (1 - prob)
                    f.write(f'"{path}",{classes[true_label]},{classes[pred_label]},{confidence:.4f}\n')
        
        print(f"? Misclassified samples saved to {misclassified_file}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60 + "\n")
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "specificity": specificity,
        "sensitivity": sensitivity
    }


# =========================
# Main Testing Script
# =========================

def main():
    # Configuration (MATCH TRAINING CONFIG)
    ROOT_DIR = "/home/23ucs623/accident/NiAD_Large_Videos"
    CHECKPOINT_PATH = "best_c3d.pth"
    BATCH_SIZE = 8  # Same as training
    DEVICE = torch.device("cuda:6")
    NUM_WORKERS = 4
    OUTPUT_DIR = "test_results_c3d"
    
    print("\n" + "="*60)
    print("C3D MODEL TESTING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Dataset: {ROOT_DIR}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n? ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please ensure the model has been trained and checkpoint exists.")
        return
    
    # Load test dataset (C3D uses 16 frames)
    test_dataset = NiADVideoDataset(
        root_dir=ROOT_DIR,
        phase="Testing",
        frames_per_clip=16  # MUST match training
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\nTotal test samples: {len(test_dataset)}")
    print(f"Total test batches: {len(test_loader)}")
    
    # Load model
    model = load_model(CHECKPOINT_PATH, DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Run testing
    results = test_model(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        save_results=True,
        output_dir=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()