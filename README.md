# Conditional Generative Adversarial Networks (cGAN)

![generated_image_set](https://github.com/LeoTheG/conditional-gan-handwriting/assets/6187214/71451e17-bcdc-4b04-988b-64d1537613b7)

## Training

`python3 src/train_model.py`

Will create models `model/generator.pth` and `model/discriminator.pth`

## Running

`python3 src/main.py`

Generates one image which is a collection of 10 digits in `generated-images/`

`python3 src/generate_one.py 6`

Generates one image of one digit in `generated-images/`

## Explanation

### GANs (Generative Adversarial Networks):

Architecture: Consists of a generator (G) and a discriminator (D). The generator produces fake samples, and the discriminator tries to differentiate between real and fake samples.
Input to Generator: The generator typically receives random noise as input and outputs fake samples.
Purpose: GANs aim to generate data that is indistinguishable from real data.
Control: In vanilla GANs, you don't have control over the specifics of the generated output. For instance, in the context of image generation, you can't specify attributes like "generate an image of a cat" or "generate an image of a dog". The output depends on the random noise input and the current state of the generator.

### cGANs (Conditional Generative Adversarial Networks):

Architecture: Like GANs, cGANs also have a generator and a discriminator. However, both the generator and the discriminator receive additional conditioning information.
Input to Generator: Along with random noise, the generator receives conditioning data. This conditioning data informs or "guides" the type of output the generator should produce.
Purpose: cGANs aim to generate data that not only resembles real data but also aligns with specific conditions or attributes provided as input.
Control: The main advantage of cGANs over vanilla GANs is control. You can guide the generation process by specifying desired attributes. For instance, in the image generation context, you can specify "generate an image of a cat" or "generate an image of a dog" by providing the corresponding label as the conditioning input.
In summary, while both GANs and cGANs aim to generate data, cGANs provide a mechanism to control or guide the generation process using external conditioning information. This makes cGANs especially useful in scenarios where you want the generated output to have specific characteristics.

### Further notes

A file with a .pth extension typically contains a serialized PyTorch state dictionary. A PyTorch state dictionary is a Python dictionary that contains the state of a PyTorch model, including the model's weights, biases, and other parameters.
