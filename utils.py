import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn


# image management

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prep_img(imagename: str, size=None, mean=MEAN, std=STD):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    4) normalize
    """
    im = Image.open(imagename)
    # resize so that minimal side length is size pixels
    if size is not None:
        im = TF.resize(im, size)
    texture_tensor = TF.to_tensor(im).unsqueeze(0)  # add batch dimension
    # remove alpha channel if any
    if texture_tensor.shape[1] == 4:
        print('removing alpha channel')
        texture_tensor = texture_tensor[:, :3, :, :]
    texture_tensor = TF.normalize(texture_tensor, mean=mean, std=std)
    return texture_tensor


def denormalize(tensor: torch.Tensor, mean=MEAN, std=STD):
    """Based on torchvision.transforms.functional.normalize.
    """
    tensor = tensor.clone().squeeze()  # remove batch dimension
    mean = torch.as_tensor(mean, dtype=tensor.dtype,
                           device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype,
                          device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def to_pil(tensor: torch.Tensor):
    """Converts tensor to PIL Image, denormalizing it.
    Args: tensor (torch.Temsor): input tensor to be converted to PIL Image of torch.Size([C, H, W]).
    Returns: PIL Image: converted img.
    """
    img = tensor.clone().detach().cpu()
    img = denormalize(img).clip(0, 1)
    img = TF.to_pil_image(img)
    return img

# model management

def randomize_layer_(layer, mean=0, std=0.015):
    with torch.no_grad():
        for param in [layer.weight, layer.bias]:
            param.copy_(torch.normal(
                mean,
                std,
                param.shape
            ))

def randomize_model_(model, mean=0, std=0.015):
    """Replaces the weights of the convolutional layers of the model with a 
    white noise.

    Parameters
    ----------
    model : nn.Module
        An iterable model to set the weights thereof.
    """
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Conv2d):
            randomize_layer_(layer, mean, std)


# hooks

def register_layer_hook(layer, storage, idx):

    def save_output(self, input, output):
        # input/output are tuples of len 1 in our case:
        assert len(output) == 1
        storage[idx] = output[0]

    handle = layer.register_forward_hook(save_output)

    return handle


def register_model_hooks(
    model: nn.Module,
    target_layers: "list[int]"
):
    """Registers hooks for all target layers of the input model.
    Returns a list holding the desired outputs once the forward pass has been run,
    as well as a list of handles to unregister the hooks with.

    Parameters
    ----------
    model : nn.Module
        The model to register the hooks onto. Needs to be subscriptable.
    target_layers : list[int]
        The indices of the layers to save the output of.

    Returns
    -------
    outputs, handles
    - outputs is a list holding the desired output tensors once the forward
    pass has been run;
    - handles is a list holding the handles used to unregister the hooks (see 
    `unregister_model_hooks`).
    """
    outputs = [None for idx in target_layers]
    handles = [
        register_layer_hook(model[layer_idx], outputs, idx) for idx, layer_idx in enumerate(target_layers)
    ]
    return outputs, handles


def unregister_model_hooks(handles):
    for handle in handles:
        handle.remove()