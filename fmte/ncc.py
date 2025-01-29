import numpy as np


def exact_ncc(
    sample_variance: float,
    scaling_factor: float,
    sensitivity_bound: float,
    noise_multiplier: float,
) -> float:
    return (scaling_factor * sample_variance) / np.sqrt(
        (scaling_factor * sample_variance) ** 2
        + (sensitivity_bound * noise_multiplier) ** 2
    )


def ncc_bound(noise_multiplier: float, N_params: float) -> float:
    return np.sqrt(1.0 / (1 + N_params * noise_multiplier**2))
