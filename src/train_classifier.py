#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a binary theta‐vector classifier for **each INR method folder**
under `--base_dir` (default: output/).

Directory expected:
output/
├─ method1/          (e.g. tiny_siren)
│  ├─ NoTumor/
│  └─ Tumor/
├─ method2/
│  ├─ NoTumor/
│  └─ Tumor/
└─ method3/
   ├─ NoTumor/
   └─ Tumor/

For every methodX it saves weights to:
output/classifier_weights/methodX.pt
"""

import os, sys, argparse, random, json
from collections import Counter

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── make sure we can import src.classifiers ------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from classifiers import SmallThetaClassifier     # src/classifiers.py


# ──────────────────────── helpers ────────────────────────────────────────
class ThetaDataset(Dataset):
    def __init__(self, theta_paths, labels):
        self.items = list(zip(theta_paths, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        theta_path, label = self.items[idx]
        theta = torch.load(theta_path)
        return theta, torch.tensor(label).long()  # ✨ Bu satır önemli




def debug_dist(labels):
    print("Label distribution →", dict(Counter(labels)))


def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, total = 0., 0, 0
    with torch.no_grad():
        for thetas, lbls in loader:
            thetas, lbls = thetas.to(device), lbls.to(device)
            out   = model(thetas).squeeze()
            loss  = criterion(out, lbls.float())
            preds = (torch.sigmoid(out) > 0.5).long()
            tot_loss += loss.item() * lbls.size(0)
            correct  += (preds == lbls).sum().item()
            total    += lbls.size(0)
    return correct / total, tot_loss / total


def gather_theta_paths(method_dir):
    """
    Returns paths list and labels (0: NoTumor, 1: Tumor).
    Ignores files not ending with '_theta.pt'.
    """
    paths, labels = [], []
    for lbl_name, lbl_val in [("NoTumor", 0), ("Tumor", 1)]:
        folder = os.path.join(method_dir, lbl_name)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith("_theta.pt"):
                paths.append(os.path.join(folder, f))
                labels.append(lbl_val)
    return paths, labels


def train_for_method(method_dir, save_path, epochs=50, batch_size=150):
    paths, labels = gather_theta_paths(method_dir)
    if len(paths) == 0:
        print(f"⚠️  No theta files found in {method_dir}. Skipping.")
        return

    debug_dist(labels)

    # reproducibility
    random.seed(42); torch.manual_seed(42)

    # split data
    p_train, p_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=42)
    p_val, p_test, y_val, y_test  = train_test_split(
        p_tmp,  y_tmp,  test_size=0.5, stratify=y_tmp,   random_state=42)

    # dataloaders
    train_loader = DataLoader(ThetaDataset(p_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ThetaDataset(p_val,   y_val),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(ThetaDataset(p_test,  y_test),
                              batch_size=batch_size, shuffle=False)

    # model
    input_dim = len(torch.load(paths[0]))
    print(input_dim)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = SmallThetaClassifier(input_dim=input_dim, num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    best_val = 0.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for i, (thetas, labels) in enumerate(train_loader):
      print(f"Batch {i:02d}: {labels.tolist()}")
      if i == 4:
          break


    for ep in range(1, epochs + 1):
        model.train()
        for thetas, lbls in train_loader:

            print("Batch label breakdown:", torch.bincount(lbls))  # ← düzeltildi

            thetas, lbls = thetas.to(device), lbls.to(device)
            optimiser.zero_grad()
            logits = model(thetas).squeeze()
            loss = criterion(logits, lbls.float())
            loss.backward()
            optimiser.step()

        train_acc, _ = evaluate(model, train_loader, criterion, device)
        val_acc, _ = evaluate(model, val_loader, criterion, device)
        print(f"[{os.path.basename(method_dir)}] "
              f"Epoch {ep:03d} | TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), save_path)


    # final test
    model.load_state_dict(torch.load(save_path))
    test_acc, _ = evaluate(model, test_loader, criterion, device)
    print(f"✓ {os.path.basename(method_dir)}  TestAcc = {test_acc:.3f}  (best val {best_val:.3f})\n")


# ──────────────────────── main ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="output",
                        help="Folder with method sub-folders (default: output/)")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    method_dirs = []
    for d in sorted(os.listdir(args.base_dir)):
        full_path = os.path.join(args.base_dir, d)
        if not os.path.isdir(full_path):
            continue
        if d.startswith(".") or d == "classifier_weights":
            continue  # gizli klasörleri ve sonuç klasörünü atla
        if not any(os.path.exists(os.path.join(full_path, c)) for c in ["NoTumor", "Tumor"]):
            continue
        method_dirs.append(full_path)


    for m_dir in method_dirs:
        save_path = os.path.join(args.base_dir,
                                 "classifier_weights",
                                 f"{os.path.basename(m_dir)}.pt")
        train_for_method(m_dir, save_path, epochs=args.epochs)


if __name__ == "__main__":
    main()
