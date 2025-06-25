# src/inr_models.py

import math
import torch
import torch.nn as nn


class SmallMLP_INR(nn.Module):
    """
    A small MLP-based Implicit Neural Representation (INR).
    Architecture:
      - Linear(2 -> 64) + ReLU
      - Linear(64 -> 128) + ReLU
      - Linear(128 -> 64) + ReLU
      - Linear(64 -> 1)
    Total parameters: ~16,833
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)
        self.act = nn.ReLU()
  
        # if act_ == "relu":
        #     self.act = nn.ReLU()
        #     print("Using ReLU activation")
        # elif act_ == "tanh":
        #     self.act = nn.Tanh()
        #     print("Using Tanh activation")
        # elif act_ == "sigmoid":
        #     self.act = nn.Sigmoid()
        #     print("Using Sigmoid activation")
        # else:
        #     self.act = nn.ReLU()
        #     print(f"Using {act_} activation, which is not standard. Defaulting to ReLU.")
        
    #     self.init_weights(act=act_,method=weight_init)

    # def init_weights(self,act,method="xavier"):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             if method == "kaiming":
    #                 nn.init.kaiming_uniform_(m.weight, nonlinearity=act)
    #             elif method == "xavier":
    #                 gain = nn.init.calculate_gain(act)
    #                 nn.init.xavier_uniform_(m.weight, gain=gain)
    #             elif method == "zero":
    #                 nn.init.zeros_(m.weight)
    #             elif method == "none":
    #                 continue  
    #             else:
    #                 raise ValueError(f"Unknown weight_init method: {method}")

    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, coords):
        """
        coords: Tensor of shape (N, 2), where each row is (x, y) normalized to [0,1].
        Returns a tensor of shape (N,) representing the predicted intensity values.
        """
        x = self.act(self.fc1(coords))  
        x = self.act(self.fc2(x))       
        x = self.act(self.fc3(x)) 
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))     
        out = self.fc6(x)                 
        return out.squeeze(-1)            


class SineLayer(nn.Module):
    """
    A single layer of a SIREN (Sinusoidal Representation Network).
    Applies a linear transformation followed by a scaled sine activation.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Initialization for the first layer: uniform in [-1/in_features, 1/in_features]
                self.linear.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)
            else:
                # Initialization for subsequent layers: uniform in [-bound, bound]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        # x: Tensor of shape (N, in_features)
        return torch.sin(self.omega_0 * self.linear(x))


class TinySIREN(nn.Module):
    """
    A lightweight SIREN model for Implicit Neural Representation.
    Architecture:
      - SineLayer(2 -> 256, is_first=True, omega_0=30)
      - SineLayer(256 -> 256, is_first=False, omega_0=30)
      - SineLayer(256 -> 256, is_first=False, omega_0=30)
      - SineLayer(256 -> 256, is_first=False, omega_0=30)
      - SineLayer(256 -> 256, is_first=False, omega_0=30)
      - SineLayer(256 -> 256, is_first=False, omega_0=30)
      - Linear(256 -> 1)
    Total parameters: ~
    """

    def __init__(self, omega_0=30.0):
        super().__init__()
        self.layer1 = SineLayer(in_features=2, out_features=256, is_first=True, omega_0=omega_0)
        self.layer2 = SineLayer(in_features=256, out_features=256, is_first=False, omega_0=omega_0)
        self.layer3 = SineLayer(in_features=256, out_features=256, is_first=False, omega_0=omega_0)
        self.layer4 = SineLayer(in_features=256, out_features=256, is_first=False, omega_0=omega_0)
        self.layer5 = SineLayer(in_features=256, out_features=256, is_first=False, omega_0=omega_0)
        self.final = nn.Linear(256, 1)

    def forward(self, coords):
        """
        coords: Tensor of shape (N, 2), where each row is (x, y) normalized to [0,1].
        Returns a tensor of shape (N,) representing the predicted intensity values.
        """
        x = self.layer1(coords)  # -> (N,128)
        x = self.layer2(x)       # -> (N,256)
        x = self.layer3(x)       # -> (N,128)
        x = self.layer4(x)       # -> (N,128)
        x = self.layer5(x)       # -> (N,128)
        out = self.final(x)      # -> (N,1)
        return out.squeeze(-1)
