# gan_train.py
import torch.nn as nn
import torch

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
num_epochs = 20
batch_size = 100
learning_rate = 0.0002

class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, image_size, num_classes):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )
        
    def forward(self, x, labels):
        x = torch.cat([x, labels], 1)  # Concatenate noise and label
        return self.fc(x)