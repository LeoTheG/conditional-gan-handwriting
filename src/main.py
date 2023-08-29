import torch
import torch.nn as nn
from torchvision.utils import save_image
from generator import Generator
from discriminator import Discriminator
import random
import string

latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
# Set the number of images you want to generate
batch_size = 10  # 10 images for numbers 0 through 9
learning_rate = 0.0002

num_epochs = 100
save_epoch_interval = 10
model_directory = "models/"
num_classes = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained generator
G = Generator(
    latent_size=latent_size,
    hidden_size=hidden_size,
    image_size=image_size,
    num_classes=num_classes
).to(device)
G.load_state_dict(torch.load('models/generator.pth'))
G.eval()

# Load the trained discriminator
D = Discriminator(
    image_size=image_size,
    hidden_size=hidden_size,
    num_classes=num_classes
).to(device)
D.load_state_dict(torch.load('models/discriminator.pth'))
D.eval()

# create a random 4 letter id
id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

output_file_name = f"generated-images/generated_image_set-{id}.png"

# Generate fake images
z = torch.randn(batch_size, latent_size).to(device)  # Generate 10 latent vectors
numbers = [i for i in range(num_classes)]

labels = torch.tensor(numbers).to(device)  # Create tensor for numbers 0-9
labels_one_hot = torch.nn.functional.one_hot(labels, num_classes).float().to(device)  # One-hot encode the labels

fake_images = G(z, labels_one_hot)  # Generate the images using the Generator
outputs = D(fake_images, labels_one_hot)  # Pass the generated images to the Discriminator
save_image(fake_images.reshape(batch_size, 1, 28, 28), output_file_name, nrow=10, normalize=True)

print(f"Generated images saved to {output_file_name}")