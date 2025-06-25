import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def compute_intra_class_distance(all_thetas: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute average pairwise L2 distance among samples of each class.
    all_thetas: shape (N, D)
    labels: shape (N,)
    Returns: dict mapping class -> average intra-class distance
    """
    distances = {}
    classes = np.unique(labels)
    for cls in classes:
        idxs = np.where(labels == cls)[0]
        pts = all_thetas[idxs]  # (n_cls, D)
        n = pts.shape[0]
        if n < 2:
            distances[cls] = 0.0
            continue
        sum_d = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                sum_d += np.linalg.norm(pts[i] - pts[j])
                count += 1
        distances[cls] = sum_d / count if count > 0 else 0.0
    return distances


def compute_inter_class_distance(all_thetas: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute L2 distance between class centroids.
    all_thetas: shape (N, D)
    labels: shape (N,)
    Returns: matrix of shape (C, C) where C = number of classes
    """
    classes = np.unique(labels)
    centroids = []
    for cls in classes:
        idxs = np.where(labels == cls)[0]
        pts = all_thetas[idxs]
        centroids.append(np.mean(pts, axis=0))
    centroids = np.stack(centroids, axis=0)  # (C, D)
    C = centroids.shape[0]
    inter_dist = np.zeros((C, C))
    for i in range(C):
        for j in range(i+1, C):
            d = np.linalg.norm(centroids[i] - centroids[j])
            inter_dist[i, j] = d
            inter_dist[j, i] = d
    return inter_dist


def plot_pca_2d(all_thetas: np.ndarray, labels: np.ndarray, save_path: str = None, title: str = "PCA 2D"):
    """
    Perform PCA to 2D and scatter plot colored by labels.
    If save_path is provided, save the figure; otherwise show it.
    """
    pca = PCA(n_components=2)
    emb2 = pca.fit_transform(all_thetas)  # (N,2)
    plt.figure(figsize=(6, 6))
    classes = np.unique(labels)
    for cls in classes:
        idxs = np.where(labels == cls)[0]
        plt.scatter(emb2[idxs, 0], emb2[idxs, 1], label=f"Class {cls}", s=10)
    plt.legend()
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_tsne_2d(all_thetas: np.ndarray, labels: np.ndarray, save_path: str = None, title: str = "t-SNE 2D"):
    """
    Perform t-SNE to 2D and scatter plot colored by labels.
    If save_path is provided, save the figure; otherwise show it.
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='random', random_state=42)
    emb2 = tsne.fit_transform(all_thetas)  # (N,2)
    plt.figure(figsize=(6, 6))
    classes = np.unique(labels)
    for cls in classes:
        idxs = np.where(labels == cls)[0]
        plt.scatter(emb2[idxs, 0], emb2[idxs, 1], label=f"Class {cls}", s=10)
    plt.legend()
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def compute_silhouette(all_thetas: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score on PCA-reduced features (100 components).
    """
    pca = PCA(n_components=min(100, all_thetas.shape[1]))
    reduced = pca.fit_transform(all_thetas)  # (N,100)
    score = silhouette_score(reduced, labels)
    return score
