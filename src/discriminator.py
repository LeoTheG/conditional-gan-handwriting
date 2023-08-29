import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, image_size=784, hidden_size=256, num_classes=10):
        super(Discriminator, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(image_size + num_classes, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        x = torch.cat([x, labels], 1)  # Concatenate image and label
        return self.fc(x)
