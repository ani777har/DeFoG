import glob
import os

import torch
import numpy as np
from scipy.stats import norm
from scipy.special import beta as beta_func, betaln
from scipy.stats import beta as sp_beta
from scipy.interpolate import interp1d


def beta_pdf(x, alpha, beta):
    """Beta distribution PDF."""
    # coeff = np.exp(betaln(alpha, beta))
    # return x ** (alpha - 1) * (1 - x) ** (beta - 1) / coeff
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / beta_func(alpha, beta)


def objective_function(alpha, beta, y, t):
    """Objective function to minimize (mean squared error)."""
    y_pred = beta_pdf(t, alpha, beta)
    regularization = (alpha + beta) + (1 / alpha + 1 / beta)
    error = np.mean((y - y_pred) ** 2)
    error = error + 0.0001 * regularization
    return error


def find_decoding_error_curve(dataset_name, root=None):
    """Newest outputs/decoding_error_<dataset>/version_*/decoding_error.csv, or None."""
    root = root or os.getcwd()
    pattern = os.path.join(
        root, "outputs", f"decoding_error_{dataset_name}", "version_*",
        "decoding_error.csv",
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def build_derived_lut(csv_path, signal="soft_combined", n_lut=4097):
    """Measured P_e(t) curve -> LUT for t(tau), tau=(P_e(0)-P_e(t))/(P_e(0)-P_e(1))."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if signal not in df.columns:
        raise ValueError(
            f"{csv_path} has no column '{signal}'; available: {list(df.columns)}"
        )
    t = df["t"].to_numpy(dtype=np.float64)
    pe = df[signal].to_numpy(dtype=np.float64)

    span = pe[0] - pe[-1]
    if span <= 1e-12:
        raise ValueError(
            f"P_e is flat in '{signal}' ({csv_path}): no schedule can be derived."
        )
    tau = (pe[0] - pe) / span
    tau = np.maximum.accumulate(np.clip(tau, 0.0, 1.0))

    tau_u, idx = np.unique(tau, return_index=True)
    t_u = t[idx]

    grid = np.linspace(0.0, 1.0, n_lut)
    try:
        from scipy.interpolate import PchipInterpolator

        lut = PchipInterpolator(tau_u, t_u, extrapolate=True)(grid)
    except Exception:
        lut = np.interp(grid, tau_u, t_u)

    lut = np.maximum.accumulate(np.clip(lut, 0.0, 1.0))
    lut[0], lut[-1] = 0.0, 1.0
    return lut


class TimeDistorter:

    def __init__(
        self,
        train_distortion,
        sample_distortion,
        mu=0,
        sigma=1,
        alpha=1,
        beta=1,
        derived_curve_path=None,
        derived_signal="soft_combined",
    ):
        self.train_distortion = train_distortion  # used for sample_ft
        self.sample_distortion = sample_distortion  # used for get_ft
        self.alpha = alpha
        self.beta = beta
        print(
            f"TimeDistorter: train_distortion={train_distortion}, sample_distortion={sample_distortion}"
        )
        self.f_inv = None

        # 'derived' distortion, loaded eagerly so a bad curve fails at construction.
        self.derived_curve_path = derived_curve_path
        self.derived_signal = derived_signal
        self._derived_lut = None
        self._derived_lut_cache = {}
        if derived_curve_path is not None:
            self._derived_lut = build_derived_lut(derived_curve_path, derived_signal)
            print(
                f"TimeDistorter: derived schedule from {derived_curve_path} "
                f"(signal={derived_signal})"
            )

    def has_derived(self):
        return self._derived_lut is not None

    def _derived_table(self, device, dtype):
        key = (device, dtype)
        if key not in self._derived_lut_cache:
            self._derived_lut_cache[key] = torch.as_tensor(
                self._derived_lut, device=device, dtype=dtype
            )
        return self._derived_lut_cache[key]

    def _apply_derived(self, t):
        """Linear read of the LUT, keeping everything on-device and shaped like t."""
        if self._derived_lut is None:
            raise ValueError(
                "time_distortion='derived' needs a measured curve. Run with "
                "sample.measure_decoding_error=True first, or point "
                "sample.derived_distortion_curve at a decoding_error.csv."
            )
        lut = self._derived_table(t.device, t.dtype)
        n = lut.numel()
        pos = t.clamp(0.0, 1.0) * (n - 1)
        lo = pos.floor().long().clamp(0, n - 2)
        w = pos - lo.to(pos.dtype)
        return lut[lo] * (1.0 - w) + lut[lo + 1] * w

    def train_ft(self, batch_size, device):
        t_uniform = torch.rand((batch_size, 1), device=device)
        t_distort = self.apply_distortion(t_uniform, self.train_distortion)

        return t_distort

    def sample_ft(self, t, sample_distortion):
        t_distort = self.apply_distortion(t, sample_distortion)
        return t_distort

    def fit(self, difficulty, t_array, learning_rate=0.01, iterations=1000):
        """Fit a beta distribution to data using the method of moments."""
        alpha, beta = self.alpha, self.beta
        t_array = t_array + 1e-6  # Avoid division by zero

        for _ in range(iterations):
            y_pred = beta_pdf(t_array, alpha, beta)

            # Numerical approximation of the gradients
            epsilon = 1e-5
            grad_alpha = (
                objective_function(alpha + epsilon, beta, difficulty, t_array)
                - objective_function(alpha - epsilon, beta, difficulty, t_array)
            ) / (2 * epsilon)
            grad_beta = (
                objective_function(alpha, beta + epsilon, difficulty, t_array)
                - objective_function(alpha, beta - epsilon, difficulty, t_array)
            ) / (2 * epsilon)

            # # Add regularization gradient components
            # grad_alpha += learning_rate * (1 - 1 / alpha**2)
            # grad_beta += learning_rate * (1 + 1 / beta**2)

            # Update parameters
            alpha -= learning_rate * grad_alpha
            beta -= learning_rate * grad_beta

            alpha = min(max(0.3, alpha), 3)
            beta = min(max(0.3, beta), 3)

        y_pred = beta_pdf(t_array, alpha, beta)
        self.approximate_f_inverse(alpha, beta)

        return y_pred, alpha, beta

    def approximate_f_inverse(self, alpha, beta):
        # Generate data points
        t_values = np.linspace(0, 1, 100000)
        f_values = sp_beta.cdf(t_values, alpha, beta)

        # Sort and remove duplicates
        sorted_indices = np.argsort(f_values)
        f_values_sorted = f_values[sorted_indices]
        t_values_sorted = t_values[sorted_indices]

        # Remove duplicates
        _, unique_indices = np.unique(f_values_sorted, return_index=True)
        f_values_unique = f_values_sorted[unique_indices]
        t_values_unique = t_values_sorted[unique_indices]

        # Create the interpolation function for the inverse
        f_inv = interp1d(
            f_values_unique,
            t_values_unique,
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.f_inv = f_inv

    def apply_distortion(self, t, distortion_type):
        assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"

        if distortion_type == "identity":
            ft = t
        elif distortion_type == "cos":
            ft = (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "revcos":
            ft = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "polyinc":
            ft = t**2
        elif distortion_type == "polydec":
            ft = 2 * t - t**2
        elif distortion_type == "derived":
            ft = self._apply_derived(t)
        elif distortion_type == "beta":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        elif distortion_type == "logitnormal":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        return ft
