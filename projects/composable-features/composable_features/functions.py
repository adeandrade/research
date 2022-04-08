import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.nn import Module, Sequential


def perturb(inputs: Tensor) -> Tensor:
    return inputs + torch.empty_like(inputs).uniform_(-.5, .5)


def quantize(inputs: Tensor) -> Tensor:
    return torch.round(inputs)


def add_ninf_channel(x: Tensor) -> Tensor:
    channel = torch.full_like(x[:, :1, :, :], -torch.inf)
    new_x = torch.cat((x, channel), dim=1)

    return new_x


def mse_loss(x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    return torch.sum(torch.masked_fill(torch.square(x - y), torch.logical_not(mask), value=0)) / torch.sum(mask)


def batch_mse_loss(x: Tensor, y: Tensor) -> Tensor:
    return torch.mean(torch.reshape(torch.square(x - y), shape=(x.shape[0], -1)), dim=1)


def batch_mse_loss_masked(x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    loss = torch.masked_fill(torch.square(x - y), torch.logical_not(mask), value=0)
    loss = torch.mean(torch.reshape(loss, shape=(x.shape[0], -1)), dim=1)

    return loss


def batch_cross_entropy(predictions: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    loss = functional.cross_entropy(predictions, targets, ignore_index=ignore_index, reduction='none')
    loss = torch.mean(torch.reshape(loss, shape=(predictions.shape[0], -1)), dim=1)

    return loss


def calculate_likelihood_entropy(likelihoods: Tensor, reduce: bool = True, normalize: bool = True) -> Tensor:
    batch_size, num_channels, height, width = likelihoods.shape
    num_pixels = num_channels * height * width

    entropy = torch.reshape(likelihoods, shape=(batch_size, num_channels, -1))
    entropy = torch.log2(entropy)
    entropy = -torch.sum(entropy, dim=-1)

    if reduce:
        entropy = torch.sum(entropy, dim=1)

    if normalize:
        entropy /= num_pixels

    entropy = torch.mean(entropy, dim=0)

    return entropy


def get_nested_attribute(structure: Module, index: int, name: str) -> Tensor:
    while not hasattr(structure, name):
        assert isinstance(structure, Sequential)
        structure = structure[index]

    return getattr(structure, name)
