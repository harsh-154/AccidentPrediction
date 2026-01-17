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
# 1B. Dropout Wrapper for C3D
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

import models.c3d as c3d_mod

def build_c3d(num_classes=2, pretrained_path=None):
    """
    Build C3D model from external model file
    Args:
        num_classes: Number of output classes
        pretrained_path: Path to pretrained weights file (optional)
    """
    # Try different ways to instantiate the model
    model = None
    
    if hasattr(c3d_mod, 'C3D'):
        try:
            model = c3d_mod.C3D(num_classes=num_classes)
        except TypeError:
            model = c3d_mod.C3D()
    elif hasattr(c3d_mod, 'get_model'):
        try:
            model = c3d_mod.get_model(num_classes=num_classes)
        except TypeError:
            model = c3d_mod.get_model()
    elif hasattr(c3d_mod, 'generate_model'):
        try:
            model = c3d_mod.generate_model(num_classes=num_classes)
        except TypeError:
            model = c3d_mod.generate_model()
    else:
        raise RuntimeError("Cannot find C3D model in models.c3d module")
    
    if model is None:
        raise RuntimeError("Failed to instantiate C3D model")
    
    # Modify final FC layer BEFORE loading weights if needed
    fc_replaced = False
    if hasattr(model, 'fc') and model.fc.out_features != num_classes:
        print(f"Replacing final FC layer: {model.fc.out_features} -> {num_classes} classes")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        fc_replaced = True
    elif hasattr(model, 'fc8') and model.fc8.out_features != num_classes:
        print(f"Replacing final FC layer (fc8): {model.fc8.out_features} -> {num_classes} classes")
        in_features = model.fc8.in_features
        model.fc8 = nn.Linear(in_features, num_classes)
        fc_replaced = True
    
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print("Loading pretrained weights from:", pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        # If we replaced FC layer, exclude it from loading
        if fc_replaced:
            keys_to_remove = [k for k in new_state_dict.keys() if 'fc8' in k or (k.startswith('fc.') and 'fc6' not in k and 'fc7' not in k)]
            for k in keys_to_remove:
                print(f"Skipping pretrained weight for replaced layer: {k}")
                del new_state_dict[k]
        
        # Load weights
        try:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print("Pretrained weights loaded successfully")
            if missing_keys:
                print(f"Missing keys (newly initialized): {missing_keys}")
        except Exception as e:
            print(f"Warning: Could not load some weights: {e}")
    
    return model


# =========================
# 3. Dataset (IMPROVED with better preprocessing)
# =========================

class NiADVideoDataset(Dataset):
    def __init__(self, root_dir, phase="Training", frames_per_clip=16, augment=False):
        self.root_dir = root_dir
        self.phase = phase
        self.frames_per_clip = frames_per_clip
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
        frames = torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
        
        # IMPROVED: Normalize using mean and std (important for C3D)
        # Using ImageNet stats as commonly done for C3D
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames = (frames / 255.0 - mean) / std
        
        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        # IMPROVED: Sample frames uniformly across video instead of just taking first N
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

    def _augment_frames(self, frames):
        # Ensure numpy array
        frames = np.array(frames)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2).copy()

        # IMPROVED: More aggressive augmentation
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            frames = np.clip(frames * factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            mean = frames.mean()
            frames = np.clip((frames - mean) * factor + mean, 0, 255).astype(np.uint8)

        return frames


# =========================
# 4. Config (TRANSFER LEARNING)
# =========================

ROOT_DIR = "/home/23ucs623/accident/NiAD_Large_Videos"
PRETRAINED_WEIGHTS = "/home/23ucs623/accident/weights/c3d-pretrained.pth"
BATCH_SIZE = 8
EPOCHS = 50  # REDUCED: Transfer learning needs fewer epochs
LEARNING_RATE = 1e-4  # LOWER: Fine-tuning requires smaller learning rate
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda:6")
NUM_WORKERS = 4
PATIENCE = 15
FREEZE_LAYERS = True  # NEW: Freeze early layers for transfer learning


# =========================
# 5. Training (IMPROVED)
# =========================

def train_improved():

    print("\n[STEP 1] Building C3D Model...")
    base_model = build_c3d(num_classes=2, pretrained_path=PRETRAINED_WEIGHTS)
    
    # TRANSFER LEARNING: Freeze early convolutional layers
    if FREEZE_LAYERS and PRETRAINED_WEIGHTS:
        print("\n[TRANSFER LEARNING] Freezing early layers...")
        layers_to_freeze = ['conv1', 'conv2', 'conv3a', 'conv3b', 'conv4a', 'conv4b']
        
        for name, param in base_model.named_parameters():
            # Freeze specified conv layers
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
                print(f"  Frozen: {name}")
        
        print("\n[TRAINABLE LAYERS]")
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                print(f"  Trainable: {name}")
    
    model = C3DWithDropout(base_model, dropout_p=0.5).to(DEVICE)

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Frozen Parameters: {total - trainable:,}")
    print(f"Trainable Percentage: {100 * trainable / total:.2f}%")

    # Load datasets
    train_dataset = NiADVideoDataset(ROOT_DIR, "Training", 16, augment=True)
    val_dataset = NiADVideoDataset(ROOT_DIR, "Validation", 16, augment=False)

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
        pin_memory=True,  # IMPROVED: Enable pin_memory for faster transfer
        drop_last=True,  # IMPROVED: Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Focal loss alpha
    alpha_normal = n_acc / total_count
    alpha_acc = n_normal / total_count
    alpha = torch.tensor([alpha_normal, alpha_acc]).float().to(DEVICE)

    print("Focal Loss Alpha:", alpha.tolist())

    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # TRANSFER LEARNING: Use Adam for fine-tuning (better for small LR)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # TRANSFER LEARNING: Gentler learning rate schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_val_acc = 0.0
    patience_counter = 0

    print("\nTraining Started\n")

    for epoch in range(EPOCHS):

        # TRAIN
        model.train()
        total_train_loss = 0.0
        total_correct = 0
        batch_count = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # IMPROVED: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_train_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            batch_count += 1

        train_loss = total_train_loss / batch_count
        train_acc = 100.0 * total_correct / (batch_count * BATCH_SIZE)

        # VALIDATION
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
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
                val_total += labels.size(0)

                for li, pi in zip(labels, preds):
                    class_total[li.item()] += 1
                    if li == pi:
                        class_correct[li.item()] += 1

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_normal_acc = 100.0 * class_correct[0] / max(1, class_total[0])
        val_accident_acc = 100.0 * class_correct[1] / max(1, class_total[1])

        # TRANSFER LEARNING: ReduceLROnPlateau based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1:3d} | LR={current_lr:.6f} | "
            f"TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.2f}% | "
            f"ValLoss={val_loss:.4f} | ValAcc={val_acc:.2f}% | "
            f"Normal={val_normal_acc:.2f}% | Accident={val_accident_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, "best_c3d.pth")
            print(f"  *** New best model saved! ***")
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