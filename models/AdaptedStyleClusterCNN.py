# Define the MLP model
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptedStyleClusterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten the latent output [batch_size, 128, 4, 4] -> [batch_size, 2048]
            
            # Fully Connected Layers
            nn.Linear(in_features=2048, out_features=128),  # Adjusted in_features
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits