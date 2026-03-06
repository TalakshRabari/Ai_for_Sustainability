"""
q3_train.py - CNN Model Training & Evaluation
===============================================
Trains a ResNet18 (pretrained) for land-use classification.
Evaluates with Accuracy, F1-score, Confusion Matrix.

Usage:
    python q3_train.py

Outputs:
    - outputs/q3_confusion_matrix.png
    - outputs/q3_training_curves.png
    - outputs/q3_results.txt
    - outputs/land_use_model.pth
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# Paths
# scripts/q3_train.py  ->  data is one level up
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # AiforSustainability/
CSV_PATH   = os.path.join(BASE_DIR, "scripts", "outputs", "dataset.csv")
OUT_DIR    = os.path.join(BASE_DIR, "scripts", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
IMG_SIZE   = 128
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────
class LandUseDataset(Dataset):
    def __init__(self, df, label_encoder, transform=None):
        self.df      = df.reset_index(drop=True)
        self.le      = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.le.transform([row['label']])[0]
        return img, label

# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"  Total samples: {len(df)}")
print(f"  Label distribution:\n{df['label'].value_counts().to_string()}\n")

train_df = df[df['split'] == 'train'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)
print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")

# Encode labels
le = LabelEncoder()
le.fit(df['label'])
class_names = list(le.classes_)
n_classes   = len(class_names)
print(f"  Classes ({n_classes}): {class_names}")

# Datasets and loaders
train_dataset = LandUseDataset(train_df, le, train_transform)
test_dataset  = LandUseDataset(test_df,  le, test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# Model: ResNet18 pretrained, replace final layer
# ─────────────────────────────────────────────
print("\nBuilding ResNet18 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Replace the final fully connected layer for our number of classes
model.fc = nn.Linear(model.fc.in_features, n_classes)
model = model.to(device)

# Class weights for imbalanced data
label_counts = train_df['label'].value_counts()
weights = []
for cls in class_names:
    weights.append(1.0 / (label_counts.get(cls, 1) + 1e-6))
weights = torch.tensor(weights, dtype=torch.float32).to(device)
weights = weights / weights.sum() * n_classes

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")

train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

best_val_acc = 0.0
best_state   = None

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += len(labels)

    train_loss = total_loss / total
    train_acc  = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ── Validate ──
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += len(labels)

    val_loss = total_loss / total
    val_acc  = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    scheduler.step()
    print(f"  Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# ─────────────────────────────────────────────
# Final evaluation with best model
# ─────────────────────────────────────────────
print("\nEvaluating best model on test set...")
model.load_state_dict(best_state)
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())

all_true  = np.array(all_true)
all_preds = np.array(all_preds)

acc = accuracy_score(all_true, all_preds)
f1  = f1_score(all_true, all_preds, average='weighted', zero_division=0)
f1_macro = f1_score(all_true, all_preds, average='macro', zero_division=0)
cm  = confusion_matrix(all_true, all_preds)
report = classification_report(all_true, all_preds,
                                target_names=class_names, zero_division=0)

print(f"\n  Accuracy      : {acc:.4f}")
print(f"  F1 (weighted) : {f1:.4f}")
print(f"  F1 (macro)    : {f1_macro:.4f}")
print(f"\n  Classification Report:\n{report}")

# ─────────────────────────────────────────────
# Save results text
# ─────────────────────────────────────────────
results_path = os.path.join(OUT_DIR, "q3_results.txt")
with open(results_path, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("Land Use Classification Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model       : ResNet18 (pretrained, fine-tuned)\n")
    f.write(f"Classes     : {class_names}\n")
    f.write(f"Train size  : {len(train_df)}\n")
    f.write(f"Test size   : {len(test_df)}\n\n")
    f.write(f"Accuracy      : {acc:.4f}\n")
    f.write(f"F1 (weighted) : {f1:.4f}\n")
    f.write(f"F1 (macro)    : {f1_macro:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
print(f"  -> Results saved: {results_path}")

# ─────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────
print("\nPlotting confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
ax.set_title(f'Confusion Matrix\nAccuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}',
             fontsize=12, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "q3_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Confusion matrix saved: {cm_path}")

# ─────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────
print("Plotting training curves...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train Loss', color='steelblue')
axes[0].plot(val_losses,   label='Val Loss',   color='coral')
axes[0].set_title('Loss Curves', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([a*100 for a in train_accs], label='Train Acc', color='steelblue')
axes[1].plot([a*100 for a in val_accs],   label='Val Acc',   color='coral')
axes[1].set_title('Accuracy Curves', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('ResNet18 Training Progress', fontsize=13, fontweight='bold')
plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "q3_training_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Training curves saved: {curves_path}")

# ─────────────────────────────────────────────
# Save model
# ─────────────────────────────────────────────
model_path = os.path.join(OUT_DIR, "land_use_model.pth")
torch.save(best_state, model_path)
print(f"  -> Model saved: {model_path}")

# ─────────────────────────────────────────────
# Interpretation summary
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("RESULTS INTERPRETATION")
print("=" * 55)
print(f"  Best validation accuracy : {best_val_acc:.4f}")
print(f"  Final test accuracy      : {acc:.4f}")
print(f"  F1-score (weighted)      : {f1:.4f}")
print(f"  F1-score (macro)         : {f1_macro:.4f}")
print("\n  Confusion matrix interpretation:")
for i, cls in enumerate(class_names):
    tp = cm[i, i]
    total_true = cm[i].sum()
    recall = tp / total_true if total_true > 0 else 0
    print(f"    {cls:12s}: {tp}/{total_true} correctly classified (recall={recall:.2f})")
print("\n✓ Q3 Complete!")