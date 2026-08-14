from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd

from sim_config import ExperimentConfig


STATE_N = 0
STATE_E = 1
STATE_F = 2


def leakage(s: np.ndarray | float, a: float, b: float) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    return a * (1.0 - np.exp(-b * s_arr))


def delta_c(
    s: np.ndarray | float,
    v: np.ndarray,
    d: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    return v * a * np.exp(-b * s_arr) * (1.0 - np.exp(-b * d))


def solve_threshold(delta_p: np.ndarray, v: np.ndarray, d: np.ndarray, a: float, b: float) -> np.ndarray:
    numerator = v * a * (1.0 - np.exp(-b * d))
    safe_delta_p = np.clip(delta_p, 1e-12, None)
    ratio = np.clip(numerator / safe_delta_p, 1e-12, None)
    return np.log(ratio) / b


def sample_agent_parameters(cfg: ExperimentConfig, rng: np.random.Generator) -> pd.DataFrame:
    q_mu = -0.5 * cfg.q_sigma**2
    q = rng.lognormal(mean=q_mu, sigma=cfg.q_sigma, size=cfg.n_agents)
    q = q / q.mean()

    alpha = rng.beta(cfg.alpha_beta_a, cfg.alpha_beta_b, size=cfg.n_agents)
    low_cost = rng.random(cfg.n_agents) < cfg.low_cost_share
    c_i1 = np.where(low_cost, cfg.c_low, cfg.c_high)

    low_privacy = rng.random(cfg.n_agents) < cfg.low_privacy_share
    v_i = np.empty(cfg.n_agents)
    v_i[low_privacy] = rng.uniform(*cfg.v_low_range, size=low_privacy.sum())
    v_i[~low_privacy] = rng.uniform(*cfg.v_high_range, size=(~low_privacy).sum())

    d_i = (1.0 - alpha) * q

    p_i = rng.uniform(cfg.lambda_i_low, cfg.lambda_i_high, size=cfg.n_agents) * c_i1 * alpha * q

    v_rank = pd.Series(v_i).rank(method="average").to_numpy()
    v_rank = (v_rank - 1.0) / max(cfg.n_agents - 1, 1)
    target_threshold = cfg.threshold_base + cfg.threshold_span * v_rank
    delta_p = delta_c(target_threshold, v_i, d_i, cfg.leakage_a, cfg.leakage_b)
    p_f = p_i + delta_p
    theoretical_threshold = solve_threshold(delta_p, v_i, d_i, cfg.leakage_a, cfg.leakage_b)

    params = pd.DataFrame(
        {
            "agent_id": np.arange(cfg.n_agents, dtype=int),
            "Q_i": q,
            "alpha_i": alpha,
            "c_i1": c_i1,
            "v_i": v_i,
            "d_i": d_i,
            "P_E": p_i,
            "delta_P": delta_p,
            "P_F": p_f,
            "target_threshold": target_threshold,
            "theoretical_threshold": theoretical_threshold,
            "low_cost_type": low_cost.astype(int),
            "low_privacy_group": low_privacy.astype(int),
        }
    )
    params["pi_solo"] = compute_solo_surplus_vector(params, cfg)
    params["pi_break"] = params["pi_solo"]
    return params


def build_initial_state(
    cfg: ExperimentConfig,
    seed_share_f: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    state = np.full(cfg.n_agents, STATE_E, dtype=int)
    n_n = int(round(cfg.n_agents * cfg.initial_n_share))
    n_f = int(round(cfg.n_agents * seed_share_f))
    local_rng = rng if rng is not None else np.random.default_rng(cfg.seed + int(round(seed_share_f * 1000)))

    if n_n > 0:
        state[:n_n] = STATE_N
    if n_f > 0:
        state[n_n : n_n + n_f] = STATE_F
    local_rng.shuffle(state)
    return state


def compute_s_minus_i(state: np.ndarray, d_i: np.ndarray) -> np.ndarray:
    f_contrib = (state == STATE_F).astype(float) * d_i
    return f_contrib.sum() - f_contrib


def normalize_externality_exposure(
    s_minus_i: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
) -> np.ndarray:
    if cfg.externality_exposure_normalization == "none":
        return s_minus_i
    if cfg.externality_exposure_normalization == "total_nonessential":
        denom = float(np.sum(params["d_i"].to_numpy()))
        return s_minus_i / max(denom, 1e-12)
    raise ValueError(f"unknown externality_exposure_normalization: {cfg.externality_exposure_normalization}")


def compute_externality_boundary(params: pd.DataFrame, cfg: ExperimentConfig) -> float:
    d_i = params["d_i"].to_numpy()
    v_i = params["v_i"].to_numpy()
    total_d = float(np.sum(d_i))
    reference_exposure = cfg.deposit_threshold_quantile * total_d
    s_reference = np.maximum(reference_exposure - d_i, 0.0)
    s_reference = normalize_externality_exposure(s_reference, params, cfg)
    return float(np.sum(v_i * leakage(s_reference, cfg.leakage_a, cfg.leakage_b)))


def compute_utilities(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    d_i = params["d_i"].to_numpy()
    s_minus_i = compute_s_minus_i(state, d_i)

    v_i = params["v_i"].to_numpy()
    p_i = params["P_E"].to_numpy()
    p_f = params["P_F"].to_numpy()
    c_term = params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()

    l_base = leakage(s_minus_i, cfg.leakage_a, cfg.leakage_b)
    l_full = leakage(s_minus_i + d_i, cfg.leakage_a, cfg.leakage_b)

    return {
        "S_minus_i": s_minus_i,
        "U_N": -v_i * l_base,
        "U_E": p_i - c_term - v_i * l_base,
        "U_F": p_f - c_term - v_i * l_full,
        "delta_C": v_i * (l_full - l_base),
    }


def compute_welfare_vector(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    alpha = params["alpha_i"].to_numpy()
    q = params["Q_i"].to_numpy()
    contribution = (state == STATE_E) * alpha * q + (state == STATE_F) * q
    t_total = float(np.sum(contribution))
    revenue = market_revenue(t_total, cfg.market_revenue_p0, cfg.market_revenue_p1)
    revenue_share = np.zeros(len(state), dtype=float)
    if t_total > 1e-12:
        revenue_share = contribution / t_total * revenue
    private_cost = compute_private_cost_vector(state, params, cfg)
    externality_penalty = compute_externality_penalty_vector(state, params, cfg)
    return revenue_share - private_cost - cfg.externality_penalty_eta * externality_penalty


def compute_social_welfare(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    c_star: float | None = None,
) -> float:
    t_total = total_market_size(state, params)
    revenue = market_revenue(t_total, cfg.market_revenue_p0, cfg.market_revenue_p1)
    private_cost = compute_total_private_cost(state, params, cfg)
    externality_penalty = compute_externality_penalty(state, params, cfg)
    welfare_boundary = compute_externality_boundary(params, cfg) if c_star is None else float(c_star)
    systemic_penalty = cfg.systemic_penalty_chi * max(externality_penalty - welfare_boundary, 0.0) ** 2
    return float(revenue - private_cost - cfg.externality_penalty_eta * externality_penalty - systemic_penalty)


def compute_social_welfare_subset(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    subset_idx: np.ndarray | list[int],
    c_star: float | None = None,
) -> float:
    # Subset welfare is used as a local diagnostic. Apply the same boundary
    # correction as total welfare by allocating the systemic penalty by agent.
    welfare = compute_welfare_vector(state, params, cfg)
    if len(subset_idx) == 0:
        return 0.0
    subset = np.asarray(subset_idx, dtype=int)
    externality_penalty = compute_externality_penalty(state, params, cfg)
    welfare_boundary = compute_externality_boundary(params, cfg) if c_star is None else float(c_star)
    systemic_penalty = cfg.systemic_penalty_chi * max(externality_penalty - welfare_boundary, 0.0) ** 2
    subset_share = float(len(subset)) / max(len(state), 1)
    return float(np.sum(welfare[subset]) - subset_share * systemic_penalty)


def total_market_size(state: np.ndarray, params: pd.DataFrame) -> float:
    alpha = params["alpha_i"].to_numpy()
    q = params["Q_i"].to_numpy()
    return float(np.sum((state == STATE_E) * alpha * q + (state == STATE_F) * q))


def rhood_share_basis_vector(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    if cfg.rhood_share_basis != "effective_contribution":
        raise ValueError(
            "RHOOD base shares must use effective_contribution to remain budget balanced"
        )
    q_i = params["Q_i"].to_numpy()
    alpha_i = params["alpha_i"].to_numpy()
    return ((state == STATE_E) * alpha_i * q_i + (state == STATE_F) * q_i).astype(float, copy=False)


def market_revenue(t_total: float, p0: float, p1: float) -> float:
    return p0 * t_total + p1 * (t_total**2)


def compute_solo_surplus_vector(params: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    q_i = params["Q_i"].to_numpy()
    alpha_i = params["alpha_i"].to_numpy()
    d_i = params["d_i"].to_numpy()
    c_term = params["c_i1"].to_numpy() * alpha_i * q_i
    v_i = params["v_i"].to_numpy()

    no_sale = np.zeros(len(params), dtype=float)
    essential_sale = (
        cfg.market_revenue_p0 * (alpha_i * q_i)
        + cfg.market_revenue_p1 * (alpha_i * q_i) ** 2
        - c_term
    )
    full_sale = (
        cfg.market_revenue_p0 * q_i
        + cfg.market_revenue_p1 * q_i**2
        - c_term
        - v_i * leakage(d_i, cfg.leakage_a, cfg.leakage_b)
    )
    return np.maximum.reduce([no_sale, essential_sale, full_sale]).astype(float, copy=False)


def get_solo_surplus(params: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    if "pi_solo" in params.columns:
        return params["pi_solo"].to_numpy(dtype=float)
    if "pi_break" in params.columns:
        return params["pi_break"].to_numpy(dtype=float)
    return compute_solo_surplus_vector(params, cfg)


def compute_total_private_cost(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> float:
    return float(np.sum(compute_private_cost_vector(state, params, cfg)))


def compute_private_cost_vector(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    utility_block = compute_utilities(state, params, cfg)
    s_minus_i = utility_block["S_minus_i"]
    d_i = params["d_i"].to_numpy()
    v_i = params["v_i"].to_numpy()
    c_term = params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()

    l_base = leakage(s_minus_i, cfg.leakage_a, cfg.leakage_b)
    l_full = leakage(s_minus_i + d_i, cfg.leakage_a, cfg.leakage_b)
    private_cost = np.where(
        state == STATE_F,
        c_term + v_i * l_full,
        c_term + v_i * l_base,
    )
    private_cost = np.where(state == STATE_N, v_i * l_base, private_cost)
    return private_cost.astype(float, copy=False)


def compute_externality_penalty_vector(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    utility_block = compute_utilities(state, params, cfg)
    s_minus_i = normalize_externality_exposure(utility_block["S_minus_i"], params, cfg)
    v_i = params["v_i"].to_numpy()
    return (v_i * leakage(s_minus_i, cfg.leakage_a, cfg.leakage_b)).astype(float, copy=False)


def compute_externality_penalty(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> float:
    return float(np.sum(compute_externality_penalty_vector(state, params, cfg)))


def compute_total_realized_cost(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> float:
    return float(
        np.sum(
            compute_realized_cost_vector(
                state,
                params,
                cfg,
                full_sharing_externality_multiplier=cfg.full_sharing_externality_multiplier,
            )
        )
    )


def compute_realized_cost_vector(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    full_sharing_externality_multiplier: float = 1.0,
) -> np.ndarray:
    utility_block = compute_utilities(state, params, cfg)
    s_minus_i = utility_block["S_minus_i"]
    d_i = params["d_i"].to_numpy()
    v_i = params["v_i"].to_numpy()
    c_term = params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()

    l_base = leakage(s_minus_i, cfg.leakage_a, cfg.leakage_b)
    l_full = leakage(s_minus_i + d_i, cfg.leakage_a, cfg.leakage_b)
    realized_cost = np.where(
        state == STATE_F,
        c_term + full_sharing_externality_multiplier * v_i * l_full,
        c_term + v_i * l_base,
    )
    realized_cost = np.where(state == STATE_N, v_i * l_base, realized_cost)
    return realized_cost.astype(float, copy=False)


def compute_rhood_utility_for_action(
    state: np.ndarray,
    idx: int,
    action: int,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    beta: float,
    core_idx: np.ndarray,
) -> float:
    cf_state = state.copy()
    cf_state[idx] = action

    utility_block = compute_utilities(cf_state, params, cfg)
    s_minus_i = utility_block["S_minus_i"]
    d_i = params["d_i"].to_numpy()
    q_i = params["Q_i"].to_numpy()
    alpha_i = params["alpha_i"].to_numpy()
    v_i = params["v_i"].to_numpy()
    c_term = params["c_i1"].to_numpy() * alpha_i * q_i

    l_base = leakage(s_minus_i, cfg.leakage_a, cfg.leakage_b)
    l_full = leakage(s_minus_i + d_i, cfg.leakage_a, cfg.leakage_b)
    realized_cost = np.where(cf_state == STATE_F, c_term + v_i * l_full, c_term + v_i * l_base)
    realized_cost = np.where(cf_state == STATE_N, v_i * l_base, realized_cost)

    if action == STATE_N:
        return float(-v_i[idx] * l_base[idx])

    t_total = total_market_size(cf_state, params)
    revenue = market_revenue(t_total, cfg.market_revenue_p0, cfg.market_revenue_p1)
    # Redistribution applies only to cooperative participants. Non-participants
    # receive the primitive outside option and therefore do not enter either the
    # compensated cost pool or the core-bonus denominator.
    active_mask = cf_state != STATE_N
    active_core_idx = core_idx[np.isin(core_idx, np.flatnonzero(active_mask))]
    c_total = float(np.sum(realized_cost[active_mask]))
    k = (
        beta * (revenue - c_total) / len(active_core_idx)
        if len(active_core_idx) > 0
        else 0.0
    )

    share_basis = rhood_share_basis_vector(cf_state, params, cfg)
    contribution = share_basis[idx]
    base_share = 0.0 if t_total <= 1e-12 else (1.0 - beta) * contribution / t_total * revenue
    compensation = beta * realized_cost[idx]
    bonus = k if idx in set(active_core_idx.tolist()) else 0.0
    transfer = base_share + compensation + bonus
    return float(transfer - realized_cost[idx])


def compute_utilities_with_rhood(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    beta: float,
) -> Dict[str, np.ndarray]:
    utility_block = compute_utilities(state, params, cfg)
    n_core = max(1, int(round(len(params) * cfg.core_share)))
    core_idx = params.nlargest(n_core, "v_i")["agent_id"].to_numpy(dtype=int)

    u_n = np.zeros(len(state), dtype=float)
    u_i = np.zeros(len(state), dtype=float)
    u_f = np.zeros(len(state), dtype=float)
    for idx in range(len(state)):
        u_n[idx] = compute_rhood_utility_for_action(state, idx, STATE_N, params, cfg, beta, core_idx)
        u_i[idx] = compute_rhood_utility_for_action(state, idx, STATE_E, params, cfg, beta, core_idx)
        u_f[idx] = compute_rhood_utility_for_action(state, idx, STATE_F, params, cfg, beta, core_idx)

    utility_block["U_N_rhood"] = u_n
    utility_block["U_E_rhood"] = u_i
    utility_block["U_F_rhood"] = u_f
    utility_block["beta_rhood"] = np.full(len(state), beta, dtype=float)
    utility_block["core_indicator"] = np.isin(np.arange(len(state)), core_idx).astype(int)
    return utility_block


def compute_utilities_with_deposit(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    deposit_vector: np.ndarray,
    c_star: float,
) -> Dict[str, np.ndarray]:
    utility_block = compute_utilities(state, params, cfg)
    current_c_ext = compute_externality_total(state, params, cfg)

    u_f_dep = utility_block["U_F"].copy()
    pivot_vec = np.zeros(len(state), dtype=float)
    for idx in range(len(state)):
        if deposit_vector[idx] <= 0.0:
            continue
        cf_state = state.copy()
        cf_state[idx] = STATE_F
        cf_c_ext = compute_externality_total(cf_state, params, cfg)
        pivot = float(current_c_ext <= c_star < cf_c_ext)
        pivot_vec[idx] = pivot
        u_f_dep[idx] = utility_block["U_F"][idx] - pivot * deposit_vector[idx]

    utility_block["U_F_dep"] = u_f_dep
    utility_block["pivot_vector"] = pivot_vec
    utility_block["current_c_ext"] = np.full(len(state), current_c_ext, dtype=float)
    return utility_block


def compute_utilities_with_rhood_deposit(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    beta: float,
    deposit_vector: np.ndarray,
    c_star: float,
) -> Dict[str, np.ndarray]:
    utility_block = compute_utilities_with_rhood(state, params, cfg, beta=beta)
    current_c_ext = compute_externality_total(state, params, cfg)

    u_f_rhood_dep = utility_block["U_F_rhood"].copy()
    pivot_vec = np.zeros(len(state), dtype=float)
    for idx in range(len(state)):
        if deposit_vector[idx] <= 0.0:
            continue
        cf_state = state.copy()
        cf_state[idx] = STATE_F
        cf_c_ext = compute_externality_total(cf_state, params, cfg)
        pivot = float(current_c_ext <= c_star < cf_c_ext)
        pivot_vec[idx] = pivot
        u_f_rhood_dep[idx] = utility_block["U_F_rhood"][idx] - pivot * deposit_vector[idx]

    utility_block["U_F_rhood_dep"] = u_f_rhood_dep
    utility_block["pivot_vector"] = pivot_vec
    utility_block["current_c_ext"] = np.full(len(state), current_c_ext, dtype=float)
    return utility_block


def compute_externality_total(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> float:
    return compute_externality_penalty(state, params, cfg)


def dynamic_safety_deposit_vector(
    utility_block: Dict[str, np.ndarray],
    multiple: float = 1.0,
) -> np.ndarray:
    """Compute the current-state minimum fee for the F upgrade margin."""
    gain = utility_block["U_F_rhood"] - np.maximum(
        utility_block["U_E_rhood"], utility_block["U_N_rhood"]
    )
    return float(multiple) * np.maximum(gain, 0.0)


def apply_batch_safety_threshold_update(
    state: np.ndarray,
    update_agents: np.ndarray,
    utility_block: Dict[str, np.ndarray],
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    deposit_vector: np.ndarray,
    c_star: float,
    dynamic_deposit: bool = False,
    deposit_multiple: float = 1.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    pre_state = state.copy()
    update_agents = np.asarray(update_agents, dtype=int)

    for idx in update_agents:
        pre_state[idx] = choose_action(
            current=int(state[idx]),
            u_n=float(utility_block["U_N_rhood"][idx]),
            u_i=float(utility_block["U_E_rhood"][idx]),
            u_f=float(utility_block["U_F_rhood"][idx]),
        )

    current_c_ext = compute_externality_total(state, params, cfg)
    pre_c_ext = compute_externality_total(pre_state, params, cfg)
    trigger = bool(pre_c_ext > c_star)

    next_state = pre_state.copy()
    charged_count = 0
    blocked_count = 0
    active_deposit = (
        dynamic_safety_deposit_vector(utility_block, multiple=deposit_multiple)
        if dynamic_deposit
        else deposit_vector
    )
    if trigger:
        for idx in update_agents:
            if pre_state[idx] != STATE_F or active_deposit[idx] <= 0.0:
                continue
            charged_count += 1
            revised_action = choose_action(
                current=int(state[idx]),
                u_n=float(utility_block["U_N_rhood"][idx]),
                u_i=float(utility_block["U_E_rhood"][idx]),
                u_f=float(utility_block["U_F_rhood"][idx] - active_deposit[idx]),
            )
            if revised_action != STATE_F:
                blocked_count += 1
            next_state[idx] = revised_action

    return next_state, {
        "batch_threshold_trigger": float(trigger),
        "batch_current_c_ext": current_c_ext,
        "batch_pre_c_ext": pre_c_ext,
        "batch_charged_count": float(charged_count),
        "batch_blocked_count": float(blocked_count),
        "batch_mean_active_deposit": float(np.mean(active_deposit[update_agents])) if len(update_agents) else 0.0,
    }


def choose_action(current: int, u_n: float, u_i: float, u_f: float) -> int:
    utilities = np.array([u_n, u_i, u_f], dtype=float)
    max_u = utilities.max()
    tied = np.flatnonzero(np.isclose(utilities, max_u))
    if current in tied:
        return int(current)
    for candidate in (STATE_E, STATE_N, STATE_F):
        if candidate in tied:
            return int(candidate)
    return int(tied[0])


def classify_steady_state(final_share_f: float, converged: bool, cfg: ExperimentConfig) -> str:
    if not converged:
        return "non_converged"
    if final_share_f >= cfg.high_state_threshold:
        return "high_externality"
    if final_share_f <= 0.10:
        return "low_externality"
    return "mixed_state"


def config_to_dict(cfg: ExperimentConfig) -> dict:
    data = asdict(cfg)
    data["output_dir"] = str(cfg.output_dir)
    return data
