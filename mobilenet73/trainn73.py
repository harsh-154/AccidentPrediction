import os
import cv2
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

# =========================
# 1. Focal Loss
# =========================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal

        return focal.mean()


# =========================
# 1B. Dropout Wrapper for MobileNetV2
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


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )
    print("Checkpoint saved:", filename)


# =========================
# 2. Model Build
# =========================

import sys
sys.path.append('/home/23ucc504/Accident')
from models.mobilenetv2 import get_model

def build_mobilenet(num_classes=2, pretrained_path=None):
    """Build MobileNetV2 model"""
    
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        # Detect number of classes from checkpoint
        if 'classifier.1.weight' in new_state_dict:
            pretrained_classes = new_state_dict['classifier.1.weight'].shape[0]
        elif 'classifier.weight' in new_state_dict:
            pretrained_classes = new_state_dict['classifier.weight'].shape[0]
        else:
            pretrained_classes = 600  # Default fallback
        
        print(f"Detected {pretrained_classes} classes in pretrained weights")
        
        # Build model with pretrained number of classes
        model = get_model(
            num_classes=pretrained_classes,
            sample_size=112,
            width_mult=1.0
        )
        
        # Load pretrained weights
        model.load_state_dict(new_state_dict, strict=True)
        print("Pretrained weights loaded successfully")
    else:
        # Build model without pretrained weights
        model = get_model(
            num_classes=num_classes,
            sample_size=112,
            width_mult=1.0
        )
    
    # Replace classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    
    return model


# =========================
# 3. Dataset (Modified for 3D Conv)
# =========================

class NiADVideoDataset(Dataset):
    def __init__(self, root_dir, phase="Training", frames_per_clip=16, augment=False):
        self.root_dir = root_dir
        self.phase = phase
        self.frames_per_clip = frames_per_clip  # Changed to 16 to match pretrained weights
        self.augment = augment and phase == "Training"
        self.classes = ["Normal", "Accident"]
        self.samples = []

        phase_dir = os.path.join(root_dir, phase)
        print("\nLoading", phase, "dataset from:", phase_dir)

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(phase_dir, class_name)
            if not os.path.exists(class_dir):
                print("Directory not found:", class_dir)
                continue

            files = [
                f
                for f in os.listdir(class_dir)
                if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]
            for fname in files:
                self.samples.append((os.path.join(class_dir, fname), label))

            print(" ", class_name, ":", len(files), "videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)

        if self.augment:
            frames = self._augment_frames(frames)

        # Convert to tensor and normalize
        # Shape: (frames, height, width, channels) -> (channels, frames, height, width)
        frames = torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
        frames = frames / 255.0
        return frames, label

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

        # pad missing frames
        while len(frames) < self.frames_per_clip:
            frames.append(np.zeros((112, 112, 3), dtype=np.uint8))

        return frames

    def _augment_frames(self, frames):
        # ensure numpy array
        frames = np.array(frames)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2).copy()

        # Random brightness adjustment
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * factor, 0, 255).astype(np.uint8)

        return frames


# =========================
# 4. Config
# =========================

ROOT_DIR = "/home/23ucc504/Accident/accident_prediction/NiAD_Large_Videos"
PRETRAINED_WEIGHTS = "/home/23ucc504/Accident/weights/kinetics_mobilenetv2_1.0x_RGB_16_best.pth"
BATCH_SIZE = 4  # Increased since MobileNet is lighter
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda:5")
NUM_WORKERS = 2
PATIENCE = 15
FRAMES_PER_CLIP = 16  # Match pretrained weights


# =========================
# 5. Training
# =========================

def train_improved():

    print("\n[STEP 1] Building MobileNetV2 Model...")
    base_model = build_mobilenet(num_classes=2, pretrained_path=PRETRAINED_WEIGHTS)
    model = MobileNetV2WithDropout(base_model, dropout_p=0.4).to(DEVICE)

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Total Parameters:", total)
    print("Trainable Parameters:", trainable)

    # Load datasets with 16 frames
    train_dataset = NiADVideoDataset(ROOT_DIR, "Training", FRAMES_PER_CLIP, augment=True)
    val_dataset = NiADVideoDataset(ROOT_DIR, "Validation", FRAMES_PER_CLIP, augment=False)

    train_labels = [lbl for _, lbl in train_dataset.samples]
    n_normal = train_labels.count(0)
    n_acc = train_labels.count(1)
    total_count = len(train_labels)

    print("\nClass Distribution:")
    print(" Normal:", n_normal)
    print(" Accident:", n_acc)

    # Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # Focal loss alpha
    alpha_normal = n_acc / total_count
    alpha_acc = n_normal / total_count
    alpha = torch.tensor([alpha_normal, alpha_acc]).float().to(DEVICE)

    print("Focal Loss Alpha:", alpha.tolist())

    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0.0
    patience_counter = 0

    print("\nTraining Started\n")

    for epoch in range(EPOCHS):

        # TRAIN
        model.train()
        total_train_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_train_loss / len(train_loader)
        train_acc = 100.0 * total_correct / len(train_dataset)

        # VALIDATION
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()

                for li, pi in zip(labels, preds):
                    class_total[li.item()] += 1
                    if li == pi:
                        class_correct[li.item()] += 1

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / len(val_dataset)
        val_normal_acc = 100.0 * class_correct[0] / max(1, class_total[0])
        val_accident_acc = 100.0 * class_correct[1] / max(1, class_total[1])

        scheduler.step()

        print(
            "Epoch", epoch + 1,
            "| TrainLoss=", round(train_loss, 4),
            "| TrainAcc=", round(train_acc, 2),
            "| ValLoss=", round(val_loss, 4),
            "| ValAcc=", round(val_acc, 2),
            "| NormalAcc=", round(val_normal_acc, 2),
            "| AccidentAcc=", round(val_accident_acc, 2)
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, "best_mobilenet.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining Complete")
    print("Best Validation Accuracy:", best_val_acc)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    train_improved()