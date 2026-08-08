"""
Physics-informed auxiliary losses for PRISM training.

All functions operate in the **normalized (z-score) space** that the CSV already
contains.  Two caveats:

  Wiebe ordering: ordering is preserved under monotone (z-score) transform ✓
  Wiebe shape:    uses an empirically-fitted target ratio (0.3821) measured in
                  normalized space for the LDC2025 dataset.
  Energy:         the physical IMEP = C·η_i·FM product does not survive z-score
                  normalization (cross-terms break it), so we use a linear OLS
                  approximation fitted from training data instead.

Column index conventions (matching config.py):
  ACP_COLS:    MFB10=0, MFB50=1, MFB90=2, p_max=3
  TARGET_COLS: eta_i=0, vol_eff=1, T_exh=2, IMEP=3
  INPUT_COLS:  N=0, p_int=1, EGR=2, T_int=3, p_diff=4, FM=5, EAR=6, IT=7
"""

import numpy as np
import torch
import torch.nn.functional as F

# Default column indices (mirrors config.py)
_MFB10_IDX = 0
_MFB50_IDX = 1
_MFB90_IDX = 2

_ETAI_IDX  = 0
_IMEP_IDX  = 3

_FM_IDX    = 5

# Empirical Wiebe shape ratio for LDC2025 normalized space:
# CA10_50 / CA10_90 mean ± std across training set = 0.3821 ± 0.0847
LDC2025_WIEBE_RATIO = 0.3821


# ---------------------------------------------------------------------------
# 1. Wiebe ordering constraint
# ---------------------------------------------------------------------------

def wiebe_ordering_loss(acp_hat: torch.Tensor) -> torch.Tensor:
    """
    Soft penalty for violations of the combustion ordering: MFB10 < MFB50 < MFB90.

    Because z-score is a monotone transform, the ordering is identical in
    normalized and physical space.  ReLU gives zero loss for valid orderings.

    Args:
        acp_hat: (B, 4) predicted ACP tensor [MFB10, MFB50, MFB90, p_max]

    Returns:
        Scalar mean penalty (≥ 0).
    """
    mfb10 = acp_hat[:, _MFB10_IDX]
    mfb50 = acp_hat[:, _MFB50_IDX]
    mfb90 = acp_hat[:, _MFB90_IDX]
    return (F.relu(mfb10 - mfb50) + F.relu(mfb50 - mfb90)).mean()


# ---------------------------------------------------------------------------
# 2. Wiebe shape constraint
# ---------------------------------------------------------------------------

def wiebe_shape_loss(
    acp_hat: torch.Tensor,
    target_ratio: float = LDC2025_WIEBE_RATIO,
    min_duration: float = 0.05,
) -> torch.Tensor:
    """
    Penalise deviation of CA10_50 / CA10_90 from the Wiebe-derived shape ratio.

    Physical background:
      From the Wiebe function x_b(θ) = 1 − exp(−a·((θ−θ_SOC)/Δθ)^(m+1)),
      the ratio (MFB50−MFB10)/(MFB90−MFB10) depends only on the shape
      exponent m and is roughly constant across operating conditions.
      For typical diesel combustion (m ≈ 2), the ratio ≈ 0.48 in physical space.
      In LDC2025 normalized space the empirical mean is 0.3821 (different
      because MFB10/50/90 have different z-score scales).

    Args:
        acp_hat:      (B, 4) predicted ACP tensor.
        target_ratio: target value of CA10_50 / CA10_90 in normalized space.
        min_duration: minimum CA10_90 (in normalized units) to include in loss;
                      filters rows where the burn window estimate is unreliable.

    Returns:
        Scalar MSE of the ratio deviation (≥ 0).
    """
    ca10_50 = acp_hat[:, _MFB50_IDX] - acp_hat[:, _MFB10_IDX]
    ca10_90 = acp_hat[:, _MFB90_IDX] - acp_hat[:, _MFB10_IDX]

    valid = ca10_90 > min_duration
    if valid.sum() == 0:
        return acp_hat.new_tensor(0.0)

    ratio = ca10_50[valid] / (ca10_90[valid] + 1e-6)
    return (ratio - target_ratio).pow(2).mean()


# ---------------------------------------------------------------------------
# 3. Energy consistency constraint (linearised for normalized space)
# ---------------------------------------------------------------------------

def estimate_energy_coefficients(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    fm_col:   int = _FM_IDX,
    etai_col: int = _ETAI_IDX,
    imep_col: int = _IMEP_IDX,
) -> tuple:
    """
    Fit IMEP_norm ≈ α·η_i_norm + β·FM_norm + γ via OLS on the training split.

    Because z-score normalization breaks the multiplicative physical law
    IMEP = C·η_i·FM (cross-product bias terms appear), we use the best linear
    approximation instead.  Coefficients are estimated once at training start.

    Returns:
        (alpha, beta, intercept) as Python floats.
    """
    imep = y_train[:, imep_col].numpy()
    etai = y_train[:, etai_col].numpy()
    fm   = x_train[:, fm_col].numpy()
    A    = np.column_stack([etai, fm, np.ones(len(imep))])
    coeffs, _, _, _ = np.linalg.lstsq(A, imep, rcond=None)
    alpha, beta, intercept = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return alpha, beta, intercept


def energy_consistency_loss(
    y_hat:     torch.Tensor,
    x:         torch.Tensor,
    alpha:     float,
    beta:      float,
    intercept: float,
    fm_col:    int = _FM_IDX,
    etai_col:  int = _ETAI_IDX,
    imep_col:  int = _IMEP_IDX,
) -> torch.Tensor:
    """
    Penalise deviation of IMEP_hat from the linear energy approximation:
      IMEP_hat ≈ α·η_i_hat + β·FM + γ

    Gradients flow through both imep_hat and etai_hat, creating mutual
    consistency pressure.  The FM term is an observed input — no gradient.

    Args:
        y_hat:  (B, 4) predicted targets [eta_i, vol_eff, T_exh, IMEP].
        x:      (B, 8) input features.
        alpha, beta, intercept: from estimate_energy_coefficients().

    Returns:
        Scalar MSE loss (≥ 0).
    """
    imep_hat  = y_hat[:, imep_col]
    etai_hat  = y_hat[:, etai_col]
    fm        = x[:, fm_col]
    imep_pred = alpha * etai_hat + beta * fm + intercept
    return F.mse_loss(imep_hat, imep_pred)
