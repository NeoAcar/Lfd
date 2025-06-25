# main.py

import argparse
import json
import os
import sys
import numpy as np
import torch

# -- Import project modules --
from src.data_preprocessing import process_brats_subset, process_acdc_subset
from src.train_inr import (
    train_batch_from_folder,
    train_tiny_siren_with_wt
)
from src.train_classifier import train_small_theta_classifier
from src.theta_utils import get_theta_paths_and_labels
from src.pca_analysis import (
    compute_intra_class_distance,
    compute_inter_class_distance,
    compute_silhouette,
    plot_pca_2d,
    plot_tsne_2d,
)
from src.gnn_module import (
    fit_pca_models,
    train_gnn,
)

def cmd_preprocess(args):
    # BRATS subset
    if args.brats_raw and args.brats_out:
        print("Processing BRATS subset...")
        process_brats_subset(
            raw_root=args.brats_raw,
            processed_root=args.brats_out,
            max_tumor=args.brats_max_tumor,
            max_no_tumor=args.brats_max_no_tumor
        )
    # ACDC subset
    if args.acdc_raw and args.acdc_out and args.labels_json:
        print("Processing ACDC subset...")
        process_acdc_subset(
            raw_root=args.acdc_raw,
            labels_path=args.labels_json,
            processed_root=args.acdc_out,
            max_per_class=args.acdc_max_per_class
        )

def cmd_train_inr(args):
    mode = args.mode
    data_dir = args.data_dir
    output_dir = args.output_dir
    omega_0 = args.omega_0
    num_samples = args.num_samples
    Epoch = args.Epoch
    lr_ = args.lr_

    if mode in ["small_mlp", "tiny_siren"]:
        train_batch_from_folder(
            data_dir=data_dir,
            output_dir=output_dir,
            model_type=mode,
            omega_0=omega_0,
            use_kfac=False,
            num_samples=num_samples,
            Epoch=Epoch,
            lr_=lr_
        )
    elif mode == "tiny_siren_wt":
        train_tiny_siren_with_wt(
            folder_dir=data_dir,
            output_root=output_dir,
            omega_0=omega_0,
            use_kfac=False,
            num_samples=num_samples,
            Epoch=Epoch,
            lr_=lr_

        )
    elif mode == "tiny_siren_wt_kfac":
        train_tiny_siren_with_wt(
            folder_dir=data_dir,
            output_root=output_dir,
            omega_0=omega_0,
            use_kfac=True,
            num_samples=num_samples,
            Epoch=Epoch,
            lr_=lr_

        )

def cmd_train_classifier(args):
    # Load class_to_idx mapping from JSON
    with open(args.class_to_idx, "r") as f:
        class_to_idx = json.load(f)

    theta_paths, labels = get_theta_paths_and_labels(args.theta_dir, class_to_idx)
    train_small_theta_classifier(
        theta_paths=theta_paths,
        labels=labels,
        out_path=args.output_model,
        Epoch=args.Epoch
    )

def cmd_pca(args):
    # Load all theta vectors
    # Build labels array
    with open(args.labels_file, "r") as f:
        label_map = json.load(f)  # expects {"theta_filename": label, ...} or list format
    
    # For simplicity, assume labels_file is a JSON mapping from full path to integer label
    theta_paths = []
    labels = []
    for path, label in label_map.items():
        theta_paths.append(path)
        labels.append(label)
    thetas = [torch.load(p).numpy() for p in theta_paths]
    all_thetas = np.stack(thetas, axis=0)
    labels_arr = np.array(labels)

    # Intra-class / inter-class / silhouette
    intra = compute_intra_class_distance(all_thetas, labels_arr)
    inter = compute_inter_class_distance(all_thetas, labels_arr)
    sil = compute_silhouette(all_thetas, labels_arr, n_components=100)

    print("Intra-class distances:", intra)
    print("Inter-class distance matrix:\n", inter)
    print("Silhouette score (PCA100):", sil)

    # PCA 2D plot
    plot_pca_2d(all_thetas, labels_arr, title="PCA 2D", save_path=os.path.join(args.output_dir, "pca2d.png"))
    # t-SNE 2D plot
    plot_tsne_2d(all_thetas, labels_arr, title="t-SNE 2D", save_path=os.path.join(args.output_dir, "tsne2d.png"))

def cmd_train_gnn(args):
    # Load class_to_idx mapping
    with open(args.class_to_idx, "r") as f:
        class_to_idx = json.load(f)

    train_gnn(
        graph_root=args.graph_root,
        class_to_idx=class_to_idx,
        k_list=args.k_list,
        pca_pickle=args.pca_pickle,
        split_dirs=args.split_dirs,
        output_model_path=args.output_model,
        num_classes=args.num_classes
    )

def main():
    parser = argparse.ArgumentParser(description="Main entrypoint for INR Classification Project")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: preprocess
    p_pre = subparsers.add_parser("preprocess", help="Generate processed subsets for BRATS and ACDC")
    p_pre.add_argument("--brats_raw", type=str, help="Raw BRATS2021 directory")
    p_pre.add_argument("--brats_out", type=str, help="Processed BRATS subset output directory")
    p_pre.add_argument("--brats_max_tumor", type=int, default=100)
    p_pre.add_argument("--brats_max_no_tumor", type=int, default=100)
    p_pre.add_argument("--acdc_raw", type=str, help="Raw ACDC directory")
    p_pre.add_argument("--acdc_out", type=str, help="Processed ACDC subset output directory")
    p_pre.add_argument("--labels_json", type=str, help="Path to ACDC labels.json")
    p_pre.add_argument("--acdc_max_per_class", type=int, default=40)
    p_pre.set_defaults(func=cmd_preprocess)

    # Subcommand: train_inr
    p_inr = subparsers.add_parser("train_inr", help="Train INR models and save theta vectors")
    p_inr.add_argument("--mode", choices=["small_mlp", "tiny_siren", "tiny_siren_wt", "tiny_siren_wt_kfac"], required=True)
    p_inr.add_argument("--data_dir", type=str, required=True, help="Processed data dir (e.g., data/Processed/ACDC_Subset)")
    p_inr.add_argument("--output_dir", type=str, required=True, help="Theta output root dir")
    p_inr.add_argument("--omega_0", type=float, default=30.0, help="Omega_0 for TinySIREN")
    p_inr.add_argument("--num_samples", type=int, default=30, help="Number of samples per class for training")
    p_inr.add_argument("--Epoch", type=int, default=301, help="Number of training epochs")
    p_inr.add_argument("--lr_", type=float, default=1e-3, help="Learning rate for training")
    p_inr.set_defaults(func=cmd_train_inr)

    # Subcommand: train_classifier
    p_cls = subparsers.add_parser("train_classifier", help="Train theta-based MLP classifier")
    p_cls.add_argument("--theta_dir", type=str, required=True, help="Directory of theta .pt files organized by class")
    p_cls.add_argument("--class_to_idx", type=str, required=True, help="JSON file mapping class name to index")
    p_cls.add_argument("--num_classes", type=int, required=True)
    p_cls.add_argument("--output_model", type=str, default="small_theta_classifier.pt",
                       help="Path to save the trained classifier model")
    p_cls.add_argument("--Epoch", type=int, default=100, help="Number of training epochs")
    p_cls.set_defaults(func=cmd_train_classifier)

    # Subcommand: pca_analysis
    p_pca = subparsers.add_parser("pca_analysis", help="Compute PCA/t-SNE and distance metrics")
    p_pca.add_argument("--labels_file", type=str, required=True,
                       help="JSON mapping each theta path to its numeric label")
    p_pca.add_argument("--output_dir", type=str, required=True, help="Directory to save PCA/t-SNE plots")
    p_pca.set_defaults(func=cmd_pca)

    # Subcommand: train_gnn
    p_gnn = subparsers.add_parser("train_gnn", help="Train GNN classifier on theta graphs")
    p_gnn.add_argument("--graph_root", type=str, required=True,
                       help="Base directory with split subfolders containing class subfolders of theta .pt")
    p_gnn.add_argument("--class_to_idx", type=str, required=True, help="JSON mapping class name to index")
    p_gnn.add_argument("--k_list", type=int, nargs=4, default=[16,32,32,8],
                       help="Node counts per layer for PCA reduction")
    p_gnn.add_argument("--pca_pickle", type=str, required=True, help="Path to saved PCA models pickle")
    p_gnn.add_argument("--split_dirs", type=json.loads, required=True,
                       help="JSON string mapping 'train','val','test' to their subfolder names")
    p_gnn.add_argument("--output_model", type=str, required=True, help="Path to save best GNN model .pth")
    p_gnn.add_argument("--num_classes", type=int, required=True)
    p_gnn.set_defaults(func=cmd_train_gnn)

    # Parse and dispatch
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
