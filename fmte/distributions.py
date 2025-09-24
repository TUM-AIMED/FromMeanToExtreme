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


def inverse_mse_cdf(
    gamma: float | np.ndarray,  # The probability value from the CDF, between 0 and 1
    N_dimensions: int,
    noise_multiplier: float,
    max_data_norm: float,
) -> float | np.ndarray:
    """
    Calculates eta corresponding to a given cumulative probability (gamma).

    This is the inverse of the mse_cdf function.
    """
    # 1. Re-create the exact same gamma distribution
    dist = get_mse_dist(N_dimensions, noise_multiplier, max_data_norm)

    # 2. Use the Percent Point Function (ppf) to find the corresponding eta
    eta = dist.ppf(gamma)

    return eta


def get_noise_multiplier(
    gamma: float,
    eta: float,
    N_dimensions: int,
    max_data_norm: float,
) -> float:
    """Calculates the noise multiplier that corresponds to a given CDF value.

    This function is the inverse of `mse_cdf` with respect to the
    `noise_multiplier` parameter. Given a cumulative probability `gamma`
    (the probability that the Mean Squared Error is less than or equal to `eta`),
    this function solves for the `noise_multiplier` that yields this probability.

    Note: The parameter `eta` is mathematically required for the inversion, as
    the value of `gamma` is dependent on it.

    Args:
        gamma: The target cumulative probability, i.e., the output of `mse_cdf`.
               Must be in the interval (0, 1).
        eta: The value at which the CDF is evaluated (the upper bound for the
             Mean Squared Error). Must be a non-negative number.
        N_dimensions: The number of dimensions.
        max_data_norm: The maximum L2 norm for the data.

    Returns:
        The corresponding `noise_multiplier`.

    Raises:
        ValueError: If `gamma` is not in (0, 1) or if `eta` is negative.
    """
    # if not 0 < gamma < 1:
    #     raise ValueError("gamma must be a probability in the interval (0, 1).")
    # if eta < 0:
    #     raise ValueError("eta must be a non-negative value.")
    # if max_data_norm <= 0:
    #     raise ValueError("max_data_norm must be positive.")

    # The shape parameter 'a' for the gamma distribution.
    shape = N_dimensions / 2

    # To invert the CDF, we use the Percent Point Function (PPF), which is the
    # inverse of the CDF. We use the PPF of the *standard* gamma distribution
    # (where scale=1) to find the scaled quantile.
    # The relationship is: gamma = F_standard(eta / scale, a=shape)
    # Inverting this gives: ppf_standard(gamma, a=shape) = eta / scale
    ppf_of_standard_gamma = stats.gamma.ppf(gamma, a=shape, scale=1)

    # The ppf can be zero if gamma is very close to 0, leading to division by zero.
    if ppf_of_standard_gamma <= 0:
        return np.inf

    # From the PPF relationship, we can solve for the scale parameter:
    # scale = eta / ppf_standard(gamma, a=shape)
    scale = eta / ppf_of_standard_gamma

    # The scale parameter is also defined in `get_mse_dist` as:
    # scale = 2 * (noise_multiplier * max_data_norm)**2 / N_dimensions
    # We set the two expressions for scale equal and solve for noise_multiplier.
    # 2 * (nm * mdn)**2 / N_dim = scale
    # (nm * mdn)**2 = (scale * N_dim) / 2
    # nm**2 = (scale * N_dim) / (2 * mdn**2)

    numerator = scale * N_dimensions
    denominator = 2 * (max_data_norm**2)

    noise_multiplier_squared = numerator / denominator
    noise_multiplier = np.sqrt(noise_multiplier_squared)

    return noise_multiplier


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
