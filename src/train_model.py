import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
# batch_size refers to the number of images per batch
batch_size = 100
learning_rate = 0.0002

num_epochs = 100
save_epoch_interval = 10
model_directory = "models/"
num_classes = 10

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# MNIST dataset
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device == 'cpu':
    # end program if no GPU is available
    print('No GPU found, please use a GPU to train your neural network.')
    exit()


D = Discriminator().to(device)
G = Generator(
    latent_size=latent_size,
    hidden_size=hidden_size,
    image_size=image_size,
    num_classes=num_classes
).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    for i, (images, real_labels) in enumerate(data_loader):
        batch_size = images.size(0)
        images = images.reshape(batch_size, -1).to(device)
        
        # Convert real_labels to one-hot encoding
        real_labels_one_hot = torch.nn.functional.one_hot(real_labels, num_classes).float().to(device)

        # Labels for discriminator's loss
        real_loss_labels = torch.ones(batch_size, 1).to(device)
        fake_loss_labels = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator with real images
        outputs = D(images, real_labels_one_hot)  # Provide labels here
        d_loss_real = criterion(outputs, real_loss_labels)
        real_score = outputs

        # Generate fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_labels = torch.nn.functional.one_hot(torch.randint(0, num_classes, (batch_size,)), num_classes).float().to(device)
        fake_images = G(z, fake_labels)
        
        # Train the discriminator with fake images
        outputs = D(fake_images, fake_labels)  # Provide labels here
        d_loss_fake = criterion(outputs, fake_loss_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train the generator
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z, fake_labels)  # We can reuse the fake_labels from earlier
        outputs = D(fake_images, fake_labels)  # Provide labels here
        g_loss = criterion(outputs, real_loss_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')
            if (epoch + 1) % save_epoch_interval == 0:
                # save generator and discriminator models at specified interval
                torch.save(G.state_dict(), model_directory + f'generator_epoch_{epoch+1}.pth')
                torch.save(D.state_dict(), model_directory + f'discriminator_epoch_{epoch+1}.pth')
        

# Save the trained generator model
torch.save(G.state_dict(), model_directory + 'generator.pth')
torch.save(D.state_dict(), model_directory +'discriminator.pth')