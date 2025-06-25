import os
import torch
import json
from glob import glob


def get_theta_paths_and_labels(theta_root):
    """
    Tüm alt klasörlerdeki *_theta.pt dosyalarını gezip,
    (dosya_yolu, sayısal_label) çiftlerinden oluşan iki liste döndürür.
    
    """
    class_to_idx = {"NoTumor": 0, "Tumor": 1}
    paths = []
    labels = []
    for cls, idx in class_to_idx.items():
        cls_folder = os.path.join(theta_root, cls)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            if fname.endswith("_theta.pt"):
                paths.append(os.path.join(cls_folder, fname))
                labels.append(idx)
    return paths, labels


def flatten_weights(model):
    """
    Converts a model's parameters into a single flattened 1D tensor.
    """
    return torch.cat([p.detach().view(-1).cpu() for p in model.parameters()], dim=0)


def load_theta_paths_labels_acdc(theta_root):
    """
    ACDC için .pt pathleri ve 0-4 integer label listesi döner
    """
    theta_paths = []
    labels = []
    class_names = sorted(os.listdir(theta_root))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    for cls_name in class_names:
        cls_folder = os.path.join(theta_root, cls_name)
        for fname in os.listdir(cls_folder):
            if fname.endswith("_theta.pt"):
                theta_paths.append(os.path.join(cls_folder, fname))
                labels.append(class_to_idx[cls_name])

    return theta_paths, labels


def load_theta_paths_labels_brats(theta_root):
    """
    BRATS için binary theta path ve label (0 = NoTumor, 1 = Tumor)
    """
    theta_paths = []
    labels = []
    for label_name in ["NoTumor", "Tumor"]:
        folder = os.path.join(theta_root, label_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith("_theta.pt"):
                theta_paths.append(os.path.join(folder, fname))
                labels.append(1 if label_name == "Tumor" else 0)
    return theta_paths, labels


def load_all_theta_variants(root_dir, class_to_idx):
    """
    output/ altında method klasörlerini döner, her biri için theta_paths ve labels döner.
    """
    method_folders = sorted(os.listdir(root_dir))
    all_data = []
    for method in method_folders:
        method_path = os.path.join(root_dir, method)
        if not os.path.isdir(method_path):
            continue
        theta_paths, labels = get_theta_paths_and_labels(method_path, class_to_idx)
        all_data.append((method, theta_paths, labels))
    return all_data


def debug_print_distribution(labels):
    from collections import Counter
    print("Label distribution:", Counter(labels))
