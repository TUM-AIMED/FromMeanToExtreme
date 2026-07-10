import torch
import torchvision
import numpy as np
from opacus.validators import ModuleValidator


def get_architecture(name, input_shape, flatten_dim=None):
    """Return (transform_fn, nn.Module) tuple for use as additional_layers.

    Args:
        name: Architecture name.
        input_shape: Spatial shape of input, e.g. (3, 32, 32) or (1, 28, 28).
        flatten_dim: Flattened dimension of the data sample (product of input_shape).

    Returns:
        (transform_fn, module) where transform_fn reshapes a flat tensor to the
        network's expected input, and module is the network that acts as a "norm sink".
    """
    if flatten_dim is None:
        flatten_dim = int(np.prod(input_shape))

    C, H, W = input_shape if len(input_shape) == 3 else (1, input_shape[-2], input_shape[-1])

    def make_transform(target_c, target_h, target_w):
        def transform(x):
            img = x.reshape(1, C, H, W)
            if C == 1 and target_c == 3:
                img = img.repeat(1, 3, 1, 1)
            if img.shape[-2] != target_h or img.shape[-1] != target_w:
                img = torch.nn.functional.interpolate(img, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return img
        return transform

    if name.startswith("linear_"):
        n_params = int(name.split("_")[1])
        out_features = max(1, n_params // flatten_dim)
        module = torch.nn.Sequential(
            torch.nn.Linear(flatten_dim, out_features, bias=False),
            torch.nn.AvgPool1d(out_features),
        )
        transform = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
        return transform, module

    if name == "resnet18":
        resnet = torchvision.models.resnet18(weights=None)
        resnet = ModuleValidator.fix(resnet)
        target_h = max(H, 32)
        target_w = max(W, 32)
        transform = make_transform(3, target_h, target_w)
        module = torch.nn.Sequential(resnet, torch.nn.AvgPool1d(1000))
        return transform, module

    if name == "resnet50":
        resnet = torchvision.models.resnet50(weights=None)
        resnet = ModuleValidator.fix(resnet)
        target_h = max(H, 32)
        target_w = max(W, 32)
        transform = make_transform(3, target_h, target_w)
        module = torch.nn.Sequential(resnet, torch.nn.AvgPool1d(1000))
        return transform, module

    if name == "resnet101":
        resnet = torchvision.models.resnet101(weights=None)
        resnet = ModuleValidator.fix(resnet)
        target_h = max(H, 32)
        target_w = max(W, 32)
        transform = make_transform(3, target_h, target_w)
        module = torch.nn.Sequential(resnet, torch.nn.AvgPool1d(1000))
        return transform, module

    if name == "vgg16":
        vgg = torchvision.models.vgg16(weights=None)
        vgg = ModuleValidator.fix(vgg)
        target_h = max(H, 32)
        target_w = max(W, 32)
        transform = make_transform(3, target_h, target_w)
        module = torch.nn.Sequential(vgg, torch.nn.AvgPool1d(1000))
        return transform, module

    raise ValueError(f"Unknown architecture: {name}")


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
