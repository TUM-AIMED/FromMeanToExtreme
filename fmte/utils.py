import torch
import numpy as np


def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    if std_data == 0 or np.isnan(std_data):
        print(f"Error encountered for data : {data}")
        result = data - mean_data
    else:
        result = (data - mean_data) / (std_data)
    return result


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0 / (data0.size - 1)) * np.sum(norm_data(data0) * norm_data(data1))


def torch_to_plt(img: torch.Tensor):
    if len(img.shape) == 2:
        return img.numpy()
    return img.numpy().transpose((1, 2, 0))


def image_prior(img: torch.Tensor):
    return (img.clone() - img.min()) / (img.max() - img.min())


def image_prior_numpy(img: np.array, data_range=1.0):
    return ((img.copy() - img.min()) / (img.max() - img.min())) * data_range
