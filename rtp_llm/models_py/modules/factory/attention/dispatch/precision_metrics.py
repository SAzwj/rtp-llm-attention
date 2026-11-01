"""Signal-conditioned Decode Attention precision prefilter."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DecodeGateConfig:
    """Initial H20 calibration; regenerate offline when the target changes."""

    signal_split: float = 0.5
    budget_coefficient: float = 0.09
    roundoff_coefficient: float = 2**-9
    gain_limit: float = 0.045
    cosine_limit: float = 0.998
    nrmse_limit: float = 0.20
    ulp_limit: Optional[float] = None
    calibration_id: str = "h20-qwen3-20261030-v1"

    def to_dict(self) -> dict:
        return {
            "signal_split": self.signal_split,
            "budget_coefficient": self.budget_coefficient,
            "roundoff_coefficient": self.roundoff_coefficient,
            "gain_limit": self.gain_limit,
            "cosine_limit": self.cosine_limit,
            "nrmse_limit": self.nrmse_limit,
            "ulp_limit": self.ulp_limit,
            "calibration_id": self.calibration_id,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "DecodeGateConfig":
        return cls(**value)


BF16_GATE_CONFIG = DecodeGateConfig(
    cosine_limit=0.9995,
    nrmse_limit=0.02,
    ulp_limit=2.5,
)
FP8_GATE_CONFIG = DecodeGateConfig()


def evaluate_precision(
    candidate: torch.Tensor,
    golden: torch.Tensor,
    reference_contribution: torch.Tensor,
    fp8: bool,
    config: Optional[DecodeGateConfig] = None,
) -> dict:
    if candidate.shape != golden.shape or candidate.dtype != golden.dtype:
        raise ValueError("candidate and golden output shape/dtype differ")
    if candidate.ndim != 2 or not candidate.numel():
        raise ValueError("empty attention output")
    if reference_contribution.ndim != 2:
        raise ValueError("reference contribution must be [batch, heads]")
    batch, heads = reference_contribution.shape
    if candidate.shape[0] != batch or heads <= 0:
        raise ValueError("reference contribution batch/head shape mismatch")
    if candidate.shape[1] % heads:
        raise ValueError("attention output width is not divisible by head count")

    cfg = config or (FP8_GATE_CONFIG if fp8 else BF16_GATE_CONFIG)
    a = candidate.float().reshape(batch, heads, -1)
    b = golden.float().reshape(batch, heads, -1)
    delta = a - b
    finite = torch.isfinite(a).all((-1, -2)) & torch.isfinite(b).all((-1, -2))
    finite &= torch.isfinite(reference_contribution).all(-1)

    error = delta.square().mean((-1, -2)).sqrt()
    reference = b.square().mean((-1, -2)).sqrt()
    signal = ((a.square() + b.square()).mean((-1, -2)) / 2).sqrt()
    nrmse = error / signal.clamp_min(1e-30)
    dot = (a * b).sum((-1, -2))
    a_norm = a.square().sum((-1, -2)).sqrt()
    b_norm = b.square().sum((-1, -2)).sqrt()
    cosine = dot / (a_norm * b_norm).clamp_min(1e-30)
    cosine = torch.where((a_norm == 0) & (b_norm == 0), 1.0, cosine)
    ulp = (
        delta.abs() / (torch.maximum(a.abs(), b.abs()).clamp_min(1e-30) * 2**-7)
    ).mean((-1, -2))
    gain = delta.mul(b).sum((-1, -2)).abs() / b.square().sum((-1, -2)).clamp_min(1e-30)
    contribution = reference_contribution.square().mean(-1).sqrt()
    low_signal = reference < cfg.signal_split * contribution

    budget_ok = error <= (
        cfg.budget_coefficient * contribution + cfg.roundoff_coefficient * reference
    )
    classic_ok = (cosine >= cfg.cosine_limit) & (nrmse <= cfg.nrmse_limit)
    if cfg.ulp_limit is not None:
        classic_ok &= ulp <= cfg.ulp_limit
    sequence_pass = torch.where(low_signal, budget_ok, classic_ok)
    sequence_pass &= gain <= cfg.gain_limit
    sequence_pass &= finite
    failed = (~sequence_pass).nonzero(as_tuple=False).flatten().tolist()
    passed = not failed

    return {
        "overall_pass": passed,
        "reason": None if passed else "sequence precision failure",
        "failed_sequences": failed,
        "low_signal_sequences": low_signal.nonzero(as_tuple=False).flatten().tolist(),
        "cos_sim": cosine.min().item(),
        "nrmse": nrmse.max().item(),
        "mean_ulp": ulp.mean().item(),
        "gain": gain.max().item(),
        "rms_error": error.max().item(),
        "reference_contribution": contribution.min().item(),
        "calibration_id": cfg.calibration_id,
    }
