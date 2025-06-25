# src/classifiers.py

import torch
import torch.nn as nn


class SmallThetaClassifier(nn.Module):
    """
    A small MLP classifier that takes a flattened theta vector as input.
    Architecture:
      - Linear(input_dim -> 512) + ReLU
      - Linear(512 -> 128) + ReLU
      - Linear(128 -> 32) + ReLU
      - Linear(32 -> num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, input_dim)
        Returns logits of shape (batch_size, num_classes) if num_classes > 1,
        or (batch_size, 1) if num_classes == 1 (binary case).
        """
        x = self.relu(self.fc1(x))   # -> (batch_size, 512)
        x = self.relu(self.fc2(x))   # -> (batch_size, 128)
        x = self.relu(self.fc3(x))   # -> (batch_size, 32)
        return self.fc4(x)           # -> (batch_size, num_classes)
