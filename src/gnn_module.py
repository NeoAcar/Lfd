# src/gnn_module.py

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm


def fit_pca_models(theta_dir, layer_dims, k_list, save_path="pca_models.pkl"):
    """
    Fit PCA models for each layer across all theta files in theta_dir.
    - theta_dir: root directory containing subfolders for each class, each with *_theta.pt files.
    - layer_dims: [192, 4160, 4160, 65] for TinySIREN.
    - k_list: e.g. [16, 32, 32, 8]
    - save_path: where to pickle the list of PCA models.
    Returns list of fitted PCA objects, and saves to save_path.
    """
    # Gather all theta vectors
    all_thetas = []
    for cls in sorted(os.listdir(theta_dir)):
        cls_folder = os.path.join(theta_dir, cls)
        for fname in os.listdir(cls_folder):
            if fname.endswith("_theta.pt"):
                theta = torch.load(os.path.join(cls_folder, fname)).numpy()
                all_thetas.append(theta)
    all_thetas = np.stack(all_thetas, axis=0)  # Shape = (N_total, sum(layer_dims))

    # Split each theta into layer-specific matrices
    offsets = np.cumsum([0] + layer_dims[:-1])
    layer_mats = []
    for i, dim in enumerate(layer_dims):
        start = offsets[i]
        end = start + dim
        layer_mats.append(all_thetas[:, start:end])  # shape = (N_total, layer_dims[i])

    pca_models = []
    for mat, K in zip(layer_mats, k_list):
        pca = PCA(n_components=K)
        pca.fit(mat)
        pca_models.append(pca)

    # Save PCA models
    with open(save_path, "wb") as f:
        pickle.dump(pca_models, f)

    return pca_models


def load_pca_models(pickle_path="pca_models.pkl"):
    """Load PCA models from pickle."""
    with open(pickle_path, "rb") as f:
        pca_models = pickle.load(f)
    return pca_models


def build_node_features_tiny(theta_np, pca_models, k_list):
    """
    Given a single theta vector (numpy array of length sum(layer_dims)),
    return a torch.Tensor of shape (sum(k_list), 5) representing node features:
    [pca_component, one-hot(layer_id)].
    """
    layer_dims = [192, 4160, 4160, 65]
    offsets = np.cumsum([0] + layer_dims[:-1])
    node_vals = []
    for i, (D_i, K_i, pca) in enumerate(zip(layer_dims, k_list, pca_models)):
        start = offsets[i]
        end = start + D_i
        layer_vec = theta_np[start:end].reshape(1, -1)  # shape (1, D_i)
        reduced = pca.transform(layer_vec).squeeze(0)   # shape (K_i,)
        node_vals.append(reduced)

    all_features = []
    for i, vals_i in enumerate(node_vals):
        one_hot = np.zeros(len(k_list), dtype=np.float32)
        one_hot[i] = 1.0
        for v in vals_i:
            feat = np.concatenate([[v], one_hot])  # shape (1 + 4 = 5)
            all_features.append(feat)
    x = torch.tensor(np.stack(all_features, axis=0), dtype=torch.float32)  # shape (sum(k_list), 5)
    return x


def build_edge_index_tiny(k_list):
    """
    Build a static edge_index tensor for a graph whose layers have sizes k_list.
    Connect each node in layer i to all nodes in layer i+1 (bidirectional).
    Returns a torch.LongTensor of shape (2, num_edges).
    """
    edges = [[], []]
    offsets = np.cumsum([0] + k_list[:-1])
    for i in range(len(k_list) - 1):
        s_i = offsets[i]
        e_i = s_i + k_list[i]
        s_j = offsets[i + 1]
        e_j = s_j + k_list[i + 1]
        for u in range(s_i, e_i):
            for v in range(s_j, e_j):
                edges[0].append(u)
                edges[1].append(v)
                edges[0].append(v)
                edges[1].append(u)
    return torch.tensor(edges, dtype=torch.long)


class ThetaGraphDataset(InMemoryDataset):
    """
    In-memory dataset for Theta-based graphs (TinySIREN).
    Expects a directory structure:
      theta_dir/
        ClassA/
          prefix_theta.pt
        ClassB/
    and a list of class labels mapping to numeric IDs.
    """

    def __init__(self, root, theta_dir, pca_models, k_list, class_to_idx, split="train", transform=None, pre_transform=None):
        """
        - root: path where processed dataset (data.pt) will be saved/loaded.
        - theta_dir: base directory with class subfolders containing theta .pt files.
        - pca_models: list of PCA models for each layer.
        - k_list: e.g. [16, 32, 32, 8].
        - class_to_idx: dict mapping class name → numeric label.
        - split: one of "train", "val", "test". The train/val/test split should be prepared externally;
                 this code expects under root/raw/{split}/ a directory mirroring theta_dir structure
                 with only the theta files belonging to that split.
        """
        self.theta_dir = os.path.join(theta_dir, split)
        self.pca_models = pca_models
        self.k_list = k_list
        self.class_to_idx = class_to_idx
        super().__init__(root, transform, pre_transform)
        data_path = os.path.join(self.processed_dir, f"data_{split}.pt")
        self.data, self.slices = torch.load(data_path)

    @property
    def raw_file_names(self):
        # Not used; expecting data already split in raw/{split}/ directories
        return []

    @property
    def processed_file_names(self):
        return [f"data_train.pt", f"data_val.pt", f"data_test.pt"]

    def download(self):
        pass

    def process(self):
        data_list = []
        classes = sorted([d for d in os.listdir(self.theta_dir) if os.path.isdir(os.path.join(self.theta_dir, d))])
        for cls in classes:
            cls_folder = os.path.join(self.theta_dir, cls)
            label = self.class_to_idx[cls]
            for fname in os.listdir(cls_folder):
                if fname.endswith("_theta.pt"):
                    theta = torch.load(os.path.join(cls_folder, fname)).numpy()
                    x = build_node_features_tiny(theta, self.pca_models, self.k_list)
                    edge_index = build_edge_index_tiny(self.k_list)
                    y = torch.tensor([label], dtype=torch.long)
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data_list.append(data)
        data, slices = self.collate(data_list)
        split = os.path.basename(self.theta_dir)
        torch.save((data, slices), os.path.join(self.processed_dir, f"data_{split}.pt"))


class TinyThetaGNN(nn.Module):
    """
    GNN classifier for TinySIREN-theta graphs.
    Architecture:
      - GCNConv(5 -> 64) + ReLU
      - GCNConv(64 -> 64) + ReLU
      - global_mean_pool
      - Linear(64 -> num_classes)
    """

    def __init__(self, in_channels=5, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out


def train_gnn(graph_root, class_to_idx, k_list, pca_pickle, split_dirs, output_model_path, num_classes):
    """
    Train and validate a GNN on TinySIREN theta graphs.
    - graph_root: base directory where split subdirs "train"/"val"/"test" contain class subfolders with theta .pt.
    - class_to_idx: mapping from class name to numeric label.
    - k_list: [16,32,32,8].
    - pca_pickle: path to saved PCA models.
    - split_dirs: dict with keys "train","val","test" mapping to subfolder names under graph_root.
    - output_model_path: where to save best model state_dict.
    - num_classes: number of classes (e.g., 5 for ACDC, 2 for BRATS_Subset).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pca_models = load_pca_models(pca_pickle)

    # Create datasets
    train_ds = ThetaGraphDataset(root=os.path.join(graph_root, "train_processed"),
                                 theta_dir=os.path.join(graph_root, split_dirs["train"]),
                                 pca_models=pca_models, k_list=k_list,
                                 class_to_idx=class_to_idx, split="train")
    val_ds = ThetaGraphDataset(root=os.path.join(graph_root, "val_processed"),
                               theta_dir=os.path.join(graph_root, split_dirs["val"]),
                               pca_models=pca_models, k_list=k_list,
                               class_to_idx=class_to_idx, split="val")
    test_ds = ThetaGraphDataset(root=os.path.join(graph_root, "test_processed"),
                                theta_dir=os.path.join(graph_root, split_dirs["test"]),
                                pca_models=pca_models, k_list=k_list,
                                class_to_idx=class_to_idx, split="test")

    train_loader = GeoDataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = GeoDataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = GeoDataLoader(test_ds, batch_size=8, shuffle=False)

    model = TinyThetaGNN(in_channels=5, hidden_channels=64, out_channels=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, 101):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
            total_loss += loss.item() * batch.num_graphs
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(dim=1)
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.num_graphs
        val_acc = val_correct / val_total

        print(f"[Epoch {epoch}] Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_model_path)

    # Final Test Evaluation
    model.load_state_dict(torch.load(output_model_path))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            test_correct += (preds == batch.y).sum().item()
            test_total += batch.num_graphs
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")


def evaluate_gnn(model, val_loader, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(model.conv1.weight.device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
            loss_sum += loss.item() * batch.num_graphs
    return correct / total, loss_sum / total
