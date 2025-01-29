import numpy as np
from scipy import special, stats


def get_mse_dist(
    N_dimensions: int,
    noise_multiplier: float,
    max_data_norm: float,
):
    return stats.gamma(
        N_dimensions / 2,
        scale=2 * (noise_multiplier * max_data_norm) ** 2 / N_dimensions,
        loc=0,
    )


def mse_pdf(
    eta: np.array,
    N_dimensions: int,
    noise_multiplier: float,
    max_data_norm: float,
):
    return get_mse_dist(N_dimensions, noise_multiplier, max_data_norm).pdf(eta)


def mse_cdf(
    eta: np.array,
    N_dimensions: int,
    noise_multiplier: float,
    max_data_norm: float,
):
    return get_mse_dist(N_dimensions, noise_multiplier, max_data_norm).cdf(eta)


class PSNR_Distribution(stats.rv_continuous):

    def __init__(
        self,
        N_dimensions,
        noise_multiplier,
        data_range,
        max_data_norm=None,
    ):
        super().__init__()
        self.N_dimensions = N_dimensions
        self.noise_multiplier = noise_multiplier
        self.data_range = data_range
        self.max_data_norm = max_data_norm

    def _cdf(self, psnr_eta):
        return 1.0 - special.gammainc(
            self.N_dimensions / 2,
            (self.N_dimensions * (self.data_range**2) * (10 ** (-psnr_eta / 10)))
            / (2 * self.noise_multiplier**2 * self.max_data_norm**2),
        )


def get_psnr_dist(
    N_dimensions: int,
    noise_multiplier: float,
    data_range: float,
    max_data_norm=None,
):
    return PSNR_Distribution(N_dimensions, noise_multiplier, data_range, max_data_norm)


def psnr_pdf(
    psnr_eta: np.array,
    N_dimensions: int,
    noise_multiplier: float,
    data_range: float,
    max_data_norm=None,
):
    dist = PSNR_Distribution(N_dimensions, noise_multiplier, data_range, max_data_norm)
    return dist.pdf(psnr_eta)


def psnr_cdf(
    psnr_eta: np.array,
    N_dimensions: int,
    noise_multiplier: float,
    data_range: float,
    max_data_norm=None,
):
    dist = PSNR_Distribution(N_dimensions, noise_multiplier, data_range, max_data_norm)
    return dist.cdf(psnr_eta)
