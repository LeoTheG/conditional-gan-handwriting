import torch
import torch.nn as nn
from torchvision.utils import save_image
from generator import Generator
from discriminator import Discriminator
import argparse
import random
import string

'''
modified from main.py to generate only one image
1. Modify batch_size to 1 since you want to generate only one image.
2. Use the provided input number as the label.
3. Adjust z to have a size of 1 (for the latent vector corresponding to the image).
'''

# Argument parser
parser = argparse.ArgumentParser(description="Generate a handwritten number using Control-GAN")
parser.add_argument("number", type=int, choices=range(0,10), help="The number to generate (0-9)")
args = parser.parse_args()

latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
batch_size = 1  # Only 1 image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained generator
G = Generator(
    latent_size=latent_size,
    hidden_size=hidden_size,
    image_size=image_size,
    num_classes=10  # Number of classes is 10 for digits 0-9
).to(device)
G.load_state_dict(torch.load('models/generator_epoch_100.pth'))
G.eval()

# Load the trained discriminator (though not really necessary for this task, but just to keep it consistent)
D = Discriminator(
    image_size=image_size,
    hidden_size=hidden_size,
    num_classes=10
).to(device)
D.load_state_dict(torch.load('models/discriminator_epoch_100.pth'))
D.eval()

# create a random 4 letter id
id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

output_file_name = f"generated-images/generated_image_{args.number}-{id}.png"

# Generate fake image
z = torch.randn(batch_size, latent_size).to(device)  # Generate 1 latent vector
label = torch.tensor([args.number]).to(device)  # Use the provided number as label
label_one_hot = torch.nn.functional.one_hot(label, 10).float().to(device)  # One-hot encode the label

fake_image = G(z, label_one_hot)  # Generate the image using the Generator
save_image(fake_image.reshape(batch_size, 1, 28, 28), output_file_name, normalize=True)

print(f"Generated image saved to {output_file_name}")
