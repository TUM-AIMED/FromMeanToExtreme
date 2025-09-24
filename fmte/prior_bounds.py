import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import fsolve
from functools import partial
from typing import Sequence


def dens_func_P(x):
    return norm.pdf(x)


def dens_func_Q(x, mu, p):
    return p * norm.pdf(x, loc=mu) + (1 - p) * norm.pdf(x)


def log_likelihood_ratio_func(x, mu, p):
    z = mu * x - 0.5 * mu**2
    if z > 0:
        return z + np.log(p) + np.log1p((1 / p - 1) * np.exp(-z))
    else:
        return np.log(1 + p * (np.exp(z) - 1))


def compute_moments(llr, density):
    """Compute moments up to order 5"""
    moments = [0] * 5
    for i in range(5):
        ord = i + 1
        integrand = lambda x: llr(x) ** ord * density(x)
        moments[i], _ = quad(integrand, -np.inf, np.inf)
    return np.array(moments)


def compute_cumulants(moments):
    """Compute cumulants up to order 5"""
    kappas = [0] * 5
    kappas[0] = moments[0]
    kappas[1] = moments[1] - moments[0] ** 2
    kappas[2] = moments[2] - 3 * moments[1] * moments[0] + 2 * moments[0] ** 3
    kappas[3] = (
        moments[3]
        - 4 * moments[2] * moments[0]
        - 3 * moments[1] ** 2
        + 12 * moments[1] * moments[0] ** 2
        - 6 * moments[0] ** 4
    )
    kappas[4] = (
        moments[4]
        - 5 * moments[3] * moments[0]
        - 10 * moments[2] * moments[1]
        + 20 * moments[2] * moments[0] ** 2
        + 30 * moments[1] ** 2 * moments[0]
        - 60 * moments[1] * moments[0] ** 3
        + 24 * moments[0] ** 5
    )
    return np.array(kappas)


def compute_correction_factor(x, cumulants):
    "Compute correction factor for a distribution at x given its cumulants"
    inv_sigma_n = 1.0 / np.sqrt(cumulants[1])
    kap_3 = cumulants[2]
    kap_4 = cumulants[3]
    kap_5 = cumulants[4]
    expansion = -1.0 / 6.0 * inv_sigma_n**3 * kap_3 * (x**2 - 1.0)
    expansion -= 1.0 / 24.0 * inv_sigma_n**4 * kap_4 * (
        x**3 - 3 * x
    ) + 1.0 / 72.0 * inv_sigma_n**6 * kap_3**2 * (x**5 - 10 * x**3 + 15 * x)
    expansion -= (
        1.0 / 120.0 * inv_sigma_n**5 * kap_5 * (x**4 - 6 * x**2 + 3)
        + 1.0
        / 144.0
        * inv_sigma_n**7
        * kap_3
        * kap_4
        * (x**6 - 15 * x**4 + 45 * x**2 - 15)
        + 1.0
        / 1296.0
        * inv_sigma_n**9
        * kap_3**3
        * (x**8 - 28 * x**6 + 210 * x**4 - 420 * x**2 + 105)
    )
    return expansion * norm.pdf(x)


def approx_quantile(alpha, cumulants):
    def f(x):
        return norm.cdf(x) + compute_correction_factor(x, cumulants) - alpha

    return fsolve(f, x0=norm.ppf(alpha))


def rero_bound_without_subsampling(kappa: float, sigma: float, delta: float):
    return norm.sf(norm.isf(kappa) - (delta / sigma))


def inverse_rero_bound_without_subsampling(
    gamma: float, kappa: float, delta: float
) -> float:
    """
    Solves for sigma in the rero_bound_without_subsampling function.

    This is an analytical inversion of the original function.
    """
    # A positive sigma is only possible if gamma > kappa.
    # If gamma <= kappa, the required power is unachievable.
    if gamma <= kappa:
        return float("inf")

    # From the derivation: sigma = delta / (norm.isf(kappa) - norm.isf(gamma))
    denominator = norm.isf(kappa) - norm.isf(gamma)

    # The denominator should be positive if gamma > kappa, but as an extra check:
    if denominator <= 0:
        return float("inf")

    return delta / denominator


def rero_bound_sgm(
    kappa: Sequence[float], mu: float, sampling_probability: float, N_steps: int
) -> Sequence[float]:
    """Compute ReRo bound for the Poisson-sampled Gaussian Mechanism.

    Args:
        kappa (Sequence[float]): Prior. 0 < kappa < 1.
        mu (float): Ratio between l2-sensitivity and noise scale. Equal to 1/noise multiplier.
        p (float): Sampling rate. 0<p<1
        N (int): Number of steps.

    Returns:
        (Sequence[float]): ReRo upper bound (gamma(kappa)).
    """
    if N_steps < 30:
        raise ValueError("N must be >=30")

    moments_p = compute_moments(
        partial(log_likelihood_ratio_func, mu=mu, p=sampling_probability), dens_func_P
    )
    moments_q = compute_moments(
        partial(log_likelihood_ratio_func, mu=mu, p=sampling_probability),
        partial(dens_func_Q, mu=mu, p=sampling_probability),
    )
    cumulants_p = compute_cumulants(moments_p)
    cumulants_q = compute_cumulants(moments_q)
    mu = np.sqrt(N_steps) * ((moments_q[0] - moments_p[0]) / np.sqrt(cumulants_p[1]))
    cumulants_p *= N_steps
    cumulants_q *= N_steps
    h = approx_quantile(1 - kappa, cumulants_p)
    x = h - mu
    if cumulants_p[1] != cumulants_q[1]:
        x *= np.sqrt(cumulants_p[1] / cumulants_q[1])
    normal_cdf_x = norm.cdf(x)
    corr = compute_correction_factor(x, cumulants_q)
    return 1 - np.minimum(1.0, np.maximum(0, normal_cdf_x + corr))


def rero_bound_glrt(kappa, sigma, Delta, sampling_rate, N_steps):
    assert N_steps >= 30
    moments_p = compute_moments(
        partial(log_likelihood_ratio_func, Delta=Delta, sigma=sigma, p=sampling_rate),
        partial(dens_func_P, sigma=sigma),
    )
    moments_q = compute_moments(
        partial(log_likelihood_ratio_func, Delta=Delta, sigma=sigma, p=sampling_rate),
        partial(dens_func_Q, Delta=Delta, sigma=sigma, p=sampling_rate),
    )
    cumulants_p = compute_cumulants(moments_p)
    cumulants_q = compute_cumulants(moments_q)
    mu = np.sqrt(N_steps) * ((moments_q[0] - moments_p[0]) / np.sqrt(cumulants_p[1]))
    cumulants_p *= N_steps
    cumulants_q *= N_steps
    h = approx_quantile(1 - kappa, cumulants_p)
    x = h - mu
    if cumulants_p[1] != cumulants_q[1]:
        x *= np.sqrt(cumulants_p[1] / cumulants_q[1])
    normal_cdf_x = norm.cdf(x)
    corr = compute_correction_factor(x, cumulants_q)
    return 1 - np.minimum(1.0, np.maximum(0, normal_cdf_x + corr))


def rero_bound_glrt_without_subsampling(kappa, d, delta, sigma):
    """
    Calculates the maximum power of the two one-sided tests for the
    Gaussian Mechanism in the Relaxed Threat Model (RTM), which is
    max(1 - j_GM(alpha), 1 - j_GM_inv(alpha)).


    Args:
        kappa (float or np.ndarray): The Type-I error rate(s), between 0 and 1.
        d (int): The dimensionality of the query function's output.
        delta (float): Sensitivity of the query function.
        sigma (float, optional): The standard deviation of the Gaussian noise.

    Returns:
        float or np.ndarray: The maximum achievable power for the given alpha.
    """
    mu2 = delta / sigma

    # --- Nested helper function to calculate j_GM(alpha) ---
    def _j_GM(alpha_val, d_val, mu2_val, sigma_val):
        nc = mu2_val**2
        scale = sigma_val**2
        critical_value_sq = stats.chi2.isf(alpha_val, df=d_val, scale=scale)
        beta = stats.ncx2.cdf(critical_value_sq, df=d_val, nc=nc, scale=scale)
        return beta

    # --- Nested helper function to calculate j_GM_inv(alpha) ---
    def _j_GM_inv(alpha_val, d_val, mu2_val, sigma_val):
        nc = mu2_val**2
        scale = sigma_val**2
        critical_value_sq = stats.ncx2.ppf(alpha_val, df=d_val, nc=nc, scale=scale)
        beta = stats.chi2.sf(critical_value_sq, df=d_val, scale=scale)
        return beta

    # Calculate the power for the forward test (1 - beta)
    power_forward = 1 - _j_GM(kappa, d, mu2, sigma)

    # Calculate the power for the reverse test (1 - beta)
    power_reverse = 1 - _j_GM_inv(kappa, d, mu2, sigma)

    # Return the maximum of the two power values
    return np.maximum(power_forward, power_reverse)
