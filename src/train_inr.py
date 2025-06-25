import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

from src.inr_models import SmallMLP_INR, TinySIREN

try:
    from kfac.preconditioner import KFACPreconditioner

    has_kfac = True
except ImportError:
    has_kfac = False


def flatten_weights(model):
    """
    Convert model parameters to a single flattened 1D tensor.
    """
    return torch.cat([p.detach().view(-1).cpu() for p in model.parameters()], dim=0)


def train_inr_model(coords_np, vals_np, model_type="small_mlp", save_path=None, omega_0=30, use_kfac=False,Epoch=300,lr_= 1e-4):
    """
    Train a single INR model (SmallMLP or TinySIREN) on one slice and save its flattened weights.
    coords_np: numpy array of shape (N,2)
    vals_np: numpy array of shape (N,)
    model_type: "small_mlp" or "tiny_siren"
    save_path: where to save the flattened theta (.pt)
    omega_0: frequency parameter for TinySIREN
    use_kfac: whether to use KFAC optimizer (True/False)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    coords = torch.from_numpy(coords_np).to(device)

    vals = torch.from_numpy(vals_np).to(device)


    if model_type == "small_mlp":
        model = SmallMLP_INR().to(device)
    elif model_type == "tiny_siren":
        model = TinySIREN(omega_0=omega_0).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if use_kfac:
        if not has_kfac:
            raise ImportError("KFAC library not found. Install kfac to use natural gradient optimizer.")
        optimizer = optim.Adam(model.parameters(), lr=lr_)  # veya Adam
        preconditioner = KFACPreconditioner(model)
        print(f"Using KFAC optimizer with preconditioner")

    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_)

    # if criterion_.lower() == "mse":
    #     criterion = nn.MSELoss()
    #     print(f"Using MSE Loss")
    # elif criterion_.lower() == "ce":
    #     criterion = nn.BCEWithLogitsLoss()
    #     print(f"Using BCEWithLogits Loss")
    # else:
    #     raise ValueError(f"Unsupported criterion: {criterion}")

    criterion = nn.MSELoss()
    losses = []
    for epoch in range(Epoch):
        model.train()
        optimizer.zero_grad()
        preds = model(coords)

        loss = criterion(preds, vals)
        loss.backward()
        if use_kfac:
            preconditioner.step()  # Apply KFAC preconditioner step
        optimizer.step()
        losses.append(loss.item())
        if epoch % 33 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

    theta = flatten_weights(model)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(theta, save_path)
    with torch.no_grad():
        pred = model(coords).cpu().numpy().reshape(240, 240)

    return theta,losses, pred


def train_batch_from_folder(data_dir, output_dir, model_type="small_mlp", omega_0=30, use_kfac=False, num_samples=100, Epoch=300, lr_=1e-4):
    """
    Train INR models on all slices in a folder and save thetas.
    data_dir: contains *_coords.npy and *_vals.npy for one class or combined classes.
    output_dir: where to save theta files.
    model_type: "small_mlp" or "tiny_siren"
    omega_0: frequency for TinySIREN.
    use_kfac: whether to apply KFAC optimizer
    """
    subfolders = sorted(os.listdir(data_dir))
    for sub in subfolders:
        sub_dir = os.path.join(data_dir, sub)
        output_sub = os.path.join(output_dir, sub)
        all_coords = sorted([f for f in os.listdir(sub_dir) if f.endswith("_coords.npy")])

        if len(all_coords) > num_samples:
            all_coords = random.sample(all_coords, num_samples)
        for coords_file in tqdm(all_coords):
            prefix = coords_file.replace("_coords.npy", "")
            coords_path = os.path.join(sub_dir, prefix + "_coords.npy")
            vals_path = os.path.join(sub_dir, prefix + "_vals.npy")
            coords = np.load(coords_path)
            vals = np.load(vals_path)
            save_path = os.path.join(output_sub, prefix + "_theta.pt")
            train_inr_model(coords, vals, model_type=model_type, save_path=save_path, omega_0=omega_0, use_kfac=use_kfac, Epoch=Epoch, lr_=lr_)

def train_tiny_siren_with_wt(folder_dir, output_root, omega_0=30, use_kfac=False, Epoch=300,num_samples=100):
    """
    Train TinySIREN for each slice with weight transfer within the same class.
    folder_dir: root directory containing subfolders per class with *_coords.npy and *_vals.npy
    output_root: where to save the flattened thetas per class
    omega_0: frequency parameter for TinySIREN
    use_kfac: whether to use natural gradient (KFAC)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted(os.listdir(folder_dir))
    for cls in classes:
        class_folder = os.path.join(folder_dir, cls)
        output_cls = os.path.join(output_root, cls)
        os.makedirs(output_cls, exist_ok=True)

        previous_state = None
        # take  random num_samples from each class
        prefixes = sorted([f.replace("_coords.npy", "")
                           for f in os.listdir(class_folder)
                           if f.endswith("_coords.npy")])
        sampled_prefixes = random.sample(prefixes, min(num_samples, len(prefixes)))
        for prefix in tqdm(sampled_prefixes):
            coords_np = np.load(os.path.join(class_folder, prefix + "_coords.npy"))
            vals_np = np.load(os.path.join(class_folder, prefix + "_vals.npy"))

            model = TinySIREN(omega_0=omega_0).to(device)
            if previous_state is not None:
                model.load_state_dict(previous_state)

            if use_kfac:
                if not has_kfac:
                    raise ImportError("KFAC library not found. Install kfac to use natural gradient optimizer.")
                optimizer = optim.SGD(model.parameters(), lr=1e-3)  # veya Adam
                preconditioner = KFACPreconditioner(model)

            else:
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()
            coords_t = torch.from_numpy(coords_np).to(device)
            vals_t = torch.from_numpy(vals_np).to(device)

            for epoch in range(1, Epoch):
                model.train()
                optimizer.zero_grad()
                preds = model(coords_t)
                loss = criterion(preds, vals_t)
                loss.backward()
                if use_kfac:
                    preconditioner.step()
                optimizer.step()
                if epoch % 33 == 0:
                    print(f"[{cls} - {prefix}] Epoch {epoch}, Loss={loss.item():.6f}")

            theta = flatten_weights(model)
            torch.save(theta, os.path.join(output_cls, prefix + "_theta.pt"))
            previous_state = model.state_dict()

        previous_state = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["small_mlp", "tiny_siren", "tiny_siren_wt", "tiny_siren_wt_kfac"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--omega_0", type=float, default=30.0)
    args = parser.parse_args()

    if args.mode in ["small_mlp", "tiny_siren"]:
        train_batch_from_folder(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_type=args.mode,
            omega_0=args.omega_0,
            use_kfac=False,
            num_samples=30,
            Epoch=301,
            lr_=1e-3
        )
    elif args.mode == "tiny_siren_wt":
        train_tiny_siren_with_wt(
            folder_dir=args.data_dir,
            output_root=args.output_dir,
            omega_0=args.omega_0,
            use_kfac=False,
            num_samples=30,
            Epoch=301,
            lr_=1e-4
        )
    elif args.mode == "tiny_siren_wt_kfac":
        train_tiny_siren_with_wt(
            folder_dir=args.data_dir,
            output_root=args.output_dir,
            omega_0=args.omega_0,
            use_kfac=True,
            Epoch=301,
            num_samples=30,
            lr_=1e-4
        )