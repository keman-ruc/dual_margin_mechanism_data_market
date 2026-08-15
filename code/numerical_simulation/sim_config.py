from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "theory-vs-simulation"


@dataclass
class ExperimentConfig:
    n_agents: int = 200
    n_runs: int = 36
    seed: int = 20260428

    q_sigma: float = 0.55
    alpha_beta_a: float = 2.0
    alpha_beta_b: float = 2.0

    c_low: float = 0.10
    c_high: float = 0.90
    low_cost_share: float = 0.25

    v_low_range: tuple[float, float] = (0.20, 0.80)
    v_high_range: tuple[float, float] = (1.10, 2.00)
    low_privacy_share: float = 0.50

    lambda_i_low: float = 0.98
    lambda_i_high: float = 1.16
    threshold_base: float = 4.80
    threshold_span: float = 4.80

    leakage_a: float = 1.0
    leakage_b: float = 0.5

    max_rounds: int = 100
    async_share: float = 0.05
    convergence_window: int = 5
    convergence_eps: float = 1e-3

    low_seed_share: float = 0.00
    high_seed_share: float = 0.06
    initial_n_share: float = 0.0
    seed_scan_grid: Sequence[float] = field(
        default_factory=lambda: (0.00, 0.05, 0.10, 0.20, 0.35, 0.50)
    )

    high_state_threshold: float = 0.65
    market_revenue_p0: float = 0.50
    market_revenue_p1: float = 0.10
    externality_penalty_eta: float = 1.0
    externality_exposure_normalization: str = "total_nonessential"
    systemic_penalty_chi: float = 1.05
    full_sharing_externality_multiplier: float = 15.0
    core_share: float = 0.20
    rr_feasibility_solo_multiplier: float = 80.0
    rr_share_basis: str = "effective_contribution"
    beta_grid: Sequence[float] = field(
        default_factory=lambda: tuple(round(x, 2) for x in [i * 0.05 for i in range(21)])
    )

    safety_fee_noise_sigma: float = 0.05
    safety_fee_dynamic: bool = True
    safety_fee_threshold_quantile: float = 0.52
    safety_fee_multipliers: Sequence[float] = field(
        default_factory=lambda: (0.0, 0.5, 1.0, 1.5, 2.0)
    )
    safety_fee_candidate_share_grid: Sequence[float] = field(
        default_factory=lambda: (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36)
    )

    output_dir: Path = DEFAULT_OUTPUT_DIR


def config_for_profile(profile: str, output_dir: Path | None = None) -> ExperimentConfig:
    if profile == "quick":
        cfg = ExperimentConfig(
            n_agents=200,
            n_runs=16,
            max_rounds=80,
            low_seed_share=0.02,
            high_seed_share=0.15,
        )
    elif profile == "paper":
        cfg = ExperimentConfig()
    else:
        raise ValueError(f"unknown profile: {profile}")

    if output_dir is not None:
        cfg.output_dir = output_dir
    return cfg
