import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np


def init_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def init_model(device):
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)

    vgg.to(device)
    return vgg


# Load Iamge
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    # if max(image.size) > max_size:
    #     size = max_size
    # else:
    #     size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    image = in_transform(image).unsqueeze(0)
    return image


# Getting Features
def get_features(image, model):
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',
              '28': 'conv5_1', }
    features = {}
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features


# Creating gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


# Image Conversion
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image


def transfer_style(device, vgg, content, style, steps):
    # Making content and style features
    print('Making content and style features')
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # Creating style grams
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initializing style weights
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}
    content_weight = 1
    style_weight = 1e6
    target = content.clone().requires_grad_(True).to(device)

    # Performing optimization
    optimizer = optim.Adam([target], lr=0.003)

    # Defining a loop statement from 1 to steps+1
    print('Creating new image')
    for ii in range(1, steps + 1):  # To ensure that our loop runs for the defined number of steps
        print(ii)
        # Extracting feature for our current target image
        target_features = get_features(target, vgg)
        # Calculating the content loss for the iteration
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        # Initializing style loss
        style_loss = 0
        # The style loss is the result of a combine loss from five different layer within our model.
        # For this reason we iterate through the five style features to get the error at each layer.
        for layer in style_weights:
            # Collecting the target feature for the specific layer from the target feature variable
            target_feature = target_features[layer]
            # Applying gram matrix function to our target feature
            target_gram = gram_matrix(target_feature)
            # Getting style_gram value for our style image from the style grams variable
            style_gram = style_grams[layer]
            # Calculating the layer style loss as content loss
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            # Obtaining feature dimensions
            _, d, h, w = target_feature.shape
            # Calculating total style loss
            style_loss += layer_style_loss / (d * h * w)
        # Calculating total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        # Using the optimizer to update parameters within our target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return target

