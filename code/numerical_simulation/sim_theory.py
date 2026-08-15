from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from sim_config import ExperimentConfig
from sim_model import (
    STATE_F,
    STATE_E,
    STATE_N,
    apply_batch_safety_threshold_update,
    build_initial_state,
    choose_action,
    compute_utilities,
    compute_utilities_with_rr,
    compute_externality_total,
    compute_social_welfare,
    get_solo_surplus,
    leakage,
    rr_share_basis_vector,
)


def compute_entry_margin(params: pd.DataFrame) -> np.ndarray:
    return params["P_E"].to_numpy() - (
        params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()
    )


def compute_upgrade_margin(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    utility_block = compute_utilities(state, params, cfg)
    return params["delta_P"].to_numpy() - utility_block["delta_C"]


def compute_social_benchmark_table(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    d_i_all = params["d_i"].to_numpy()
    v_i_all = params["v_i"].to_numpy()
    delta_p_all = params["delta_P"].to_numpy()

    rows = []
    for idx in range(len(params)):
        base_state = state.copy()
        base_state[idx] = STATE_E
        base_utils = compute_utilities(base_state, params, cfg)
        base_exposure = base_utils["S_minus_i"] + (base_state == STATE_F).astype(float) * d_i_all

        upgrade_state = base_state.copy()
        upgrade_state[idx] = STATE_F
        upgrade_utils = compute_utilities(upgrade_state, params, cfg)
        upgrade_exposure = upgrade_utils["S_minus_i"] + (upgrade_state == STATE_F).astype(float) * d_i_all

        externality_vec = v_i_all * (
            leakage(upgrade_exposure, cfg.leakage_a, cfg.leakage_b)
            - leakage(base_exposure, cfg.leakage_a, cfg.leakage_b)
        )
        externality_vec[idx] = 0.0
        externality_term = float(np.sum(externality_vec))

        private_required_premium = float(base_utils["delta_C"][idx])
        social_required_premium = private_required_premium + externality_term
        delta_p = float(delta_p_all[idx])
        private_margin = delta_p - private_required_premium
        social_margin = delta_p - social_required_premium

        private_upgrade_indicator = int(private_margin >= -1e-10)
        social_upgrade_indicator = int(social_margin >= -1e-10)
        excessive_upgrade_indicator = int(private_upgrade_indicator == 1 and social_upgrade_indicator == 0)
        observed_f_indicator = int(state[idx] == STATE_F)
        observed_excessive_f_indicator = int(observed_f_indicator == 1 and excessive_upgrade_indicator == 1)

        rows.append(
            {
                "agent_id": int(idx),
                "current_action": int(state[idx]),
                "delta_P": delta_p,
                "private_required_premium": private_required_premium,
                "social_externality_term": externality_term,
                "social_required_premium": social_required_premium,
                "private_margin": private_margin,
                "social_margin": social_margin,
                "private_upgrade_indicator": private_upgrade_indicator,
                "social_upgrade_indicator": social_upgrade_indicator,
                "excessive_upgrade_indicator": excessive_upgrade_indicator,
                "observed_f_indicator": observed_f_indicator,
                "observed_excessive_f_indicator": observed_excessive_f_indicator,
            }
        )

    return pd.DataFrame(rows)


def compute_high_externality_candidate(
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    compromise_share: float = 0.35,
) -> Dict[str, object]:
    compromise_count = max(1, int(round(len(params) * compromise_share)))
    compromise_idx = params.nsmallest(compromise_count, "theoretical_threshold")["agent_id"].to_numpy()

    state_h = np.full(len(params), STATE_E, dtype=int)
    state_h[compromise_idx] = STATE_F

    utility_block = compute_utilities(state_h, params, cfg)
    s_minus_i_h = utility_block["S_minus_i"]
    delta_c_h = utility_block["delta_C"]
    delta_p = params["delta_P"].to_numpy()

    c_term = params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()
    u_n = utility_block["U_N"]
    u_f = utility_block["U_F"]

    local_eq_mask = np.zeros(len(params), dtype=int)
    local_eq_mask[compromise_idx] = (
        (delta_p[compromise_idx] >= delta_c_h[compromise_idx]) &
        (u_f[compromise_idx] >= u_n[compromise_idx])
    ).astype(int)

    return {
        "compromise_idx": compromise_idx,
        "state_h": state_h,
        "S_minus_i_h": s_minus_i_h,
        "delta_C_h": delta_c_h,
        "U_N_h": u_n,
        "U_F_h": u_f,
        "local_eq_mask": local_eq_mask,
        "local_eq_rate": float(local_eq_mask[compromise_idx].mean()),
        "c_term": c_term,
    }


def build_candidate_state_from_share(
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    target_f_share: float,
) -> Dict[str, object]:
    f_count = max(1, int(round(len(params) * target_f_share)))
    selected_idx = params.nsmallest(f_count, "theoretical_threshold")["agent_id"].to_numpy()
    state = np.full(len(params), STATE_E, dtype=int)
    state[selected_idx] = STATE_F

    utility_block = compute_utilities(state, params, cfg)
    local_eq_mask = np.zeros(len(params), dtype=int)
    local_eq_mask[selected_idx] = (
        (utility_block["U_F"][selected_idx] >= utility_block["U_E"][selected_idx]) &
        (utility_block["U_F"][selected_idx] >= utility_block["U_N"][selected_idx])
    ).astype(int)

    c_ext = compute_externality_total(state, params, cfg)

    return {
        "state": state,
        "selected_idx": selected_idx,
        "local_eq_mask": local_eq_mask,
        "local_eq_rate": float(local_eq_mask[selected_idx].mean()) if len(selected_idx) > 0 else 0.0,
        "c_ext": c_ext,
        "share_f": float(np.mean(state == STATE_F)),
        "utility_block": utility_block,
    }


def total_market_size(state: np.ndarray, params: pd.DataFrame) -> float:
    alpha = params["alpha_i"].to_numpy()
    q = params["Q_i"].to_numpy()
    return float(np.sum((state == STATE_E) * alpha * q + (state == STATE_F) * q))


def total_cost(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> float:
    utility_block = compute_utilities(state, params, cfg)
    d_i = params["d_i"].to_numpy()
    c_term = params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()
    l_base = leakage(utility_block["S_minus_i"], cfg.leakage_a, cfg.leakage_b)
    l_full = leakage(utility_block["S_minus_i"] + d_i, cfg.leakage_a, cfg.leakage_b)

    total = np.where(state == 0, params["v_i"].to_numpy() * l_base, 0.0)
    total += np.where(state == STATE_E, c_term + params["v_i"].to_numpy() * l_base, 0.0)
    total += np.where(state == STATE_F, c_term + params["v_i"].to_numpy() * l_full, 0.0)
    return float(np.sum(total))


def market_revenue(t_total: float, p0: float, p1: float) -> float:
    return p0 * t_total + p1 * (t_total**2)


def compute_rr_theory(state: np.ndarray, params: pd.DataFrame, cfg: ExperimentConfig) -> Dict[str, object]:
    n_core = max(1, int(round(len(params) * cfg.core_share)))
    core_idx = params.nlargest(n_core, "v_i")["agent_id"].to_numpy()

    t_total = total_market_size(state, params)
    revenue = market_revenue(t_total, cfg.market_revenue_p0, cfg.market_revenue_p1)
    active_mask = state != STATE_N
    active_core_idx = core_idx[np.isin(core_idx, np.flatnonzero(active_mask))]
    utility_block = compute_utilities(state, params, cfg)
    d_i = params["d_i"].to_numpy()
    l_base = leakage(utility_block["S_minus_i"], cfg.leakage_a, cfg.leakage_b)
    l_full = leakage(utility_block["S_minus_i"] + d_i, cfg.leakage_a, cfg.leakage_b)
    c_term = params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()
    realized_cost = np.where(state == STATE_F, c_term + params["v_i"].to_numpy() * l_full, c_term + params["v_i"].to_numpy() * l_base)
    realized_cost = np.where(state == 0, params["v_i"].to_numpy() * l_base, realized_cost)
    c_total = float(np.sum(realized_cost[active_mask]))

    rows = []
    outside_option = get_solo_surplus(params, cfg)
    share_basis = rr_share_basis_vector(state, params, cfg)
    for idx in active_core_idx:
        base_term = (float(share_basis[idx]) / max(t_total, 1e-8)) * revenue - float(realized_cost[idx])
        denom = (revenue - c_total) / len(active_core_idx) - base_term
        numer = float(outside_option[idx]) - base_term
        beta0 = np.nan if abs(denom) < 1e-10 else numer / denom
        rows.append(
            {
                "agent_id": int(idx),
                "beta0": float(beta0),
                "base_term": float(base_term),
                "denom": float(denom),
                "numer": float(numer),
                "pi_solo": float(outside_option[idx]),
                "pi_break": float(outside_option[idx]),
            }
        )

    beta0_df = pd.DataFrame(rows)
    feasible_rate = float(
        np.mean((beta0_df["beta0"] >= 0.0) & (beta0_df["beta0"] <= 1.0))
    ) if not beta0_df.empty else 0.0

    return {
        "beta0_df": beta0_df,
        "revenue": revenue,
        "c_total": c_total,
        "feasible_rate": feasible_rate,
    }


def scan_rr_theory_curve(
    beta0_df: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    if beta0_df.empty:
        return pd.DataFrame(columns=["beta", "all_core_feasible_theory"])

    rows = []
    tol = 1e-10
    for beta in cfg.beta_grid:
        feasible_mask = []
        for row in beta0_df.itertuples(index=False):
            denom = float(row.denom)
            numer = float(row.numer)
            beta0 = float(row.beta0)
            if abs(denom) < tol:
                feasible_mask.append(numer <= tol)
            elif denom > 0.0:
                feasible_mask.append(beta >= beta0 - tol)
            else:
                feasible_mask.append(beta <= beta0 + tol)
        rows.append(
            {
                "beta": float(beta),
                "all_core_feasible_theory": int(bool(np.all(feasible_mask))),
            }
        )
    return pd.DataFrame(rows)


def scan_rr_feasibility(
    state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    theory = compute_rr_theory(state, params, cfg)
    beta0_df = theory["beta0_df"]
    if beta0_df.empty:
        return pd.DataFrame(columns=["beta", "feasible_share", "all_core_feasible", "mean_slack"])

    n_core = len(beta0_df)
    t_total = total_market_size(state, params)
    revenue = market_revenue(t_total, cfg.market_revenue_p0, cfg.market_revenue_p1)
    core_idx = beta0_df["agent_id"].astype(int).to_numpy()

    utility_block = compute_utilities(state, params, cfg)
    d_i = params["d_i"].to_numpy()
    l_base = leakage(utility_block["S_minus_i"], cfg.leakage_a, cfg.leakage_b)
    l_full = leakage(utility_block["S_minus_i"] + d_i, cfg.leakage_a, cfg.leakage_b)
    c_term = params["c_i1"].to_numpy() * params["alpha_i"].to_numpy() * params["Q_i"].to_numpy()
    realized_cost = np.where(state == STATE_F, c_term + params["v_i"].to_numpy() * l_full, c_term + params["v_i"].to_numpy() * l_base)
    realized_cost = np.where(state == 0, params["v_i"].to_numpy() * l_base, realized_cost)
    c_total = float(np.sum(realized_cost[state != STATE_N]))

    rows = []
    outside_option = get_solo_surplus(params, cfg)
    share_basis = rr_share_basis_vector(state, params, cfg)
    for beta in cfg.beta_grid:
        k = beta * (revenue - c_total) / max(n_core, 1)
        core_utilities = []
        for idx in core_idx:
            base_share = (1.0 - beta) * float(share_basis[idx]) / max(t_total, 1e-8) * revenue
            compensation = beta * float(realized_cost[idx])
            u_rr = base_share + compensation + k - float(realized_cost[idx])
            core_utilities.append(u_rr)
        core_utilities_arr = np.asarray(core_utilities, dtype=float)
        pi_solo = outside_option[core_idx]
        feasible_mask = core_utilities_arr >= pi_solo - 1e-10
        rows.append(
            {
                "beta": float(beta),
                "feasible_share": float(feasible_mask.mean()),
                "all_core_feasible": int(bool(feasible_mask.all())),
                "mean_slack": float(np.mean(core_utilities_arr - pi_solo)),
            }
        )
    return pd.DataFrame(rows)


def compute_safety_fee_theory(
    candidate_state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    c_star: float,
    beta: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    utility_block = compute_utilities_with_rr(candidate_state, params, cfg, beta=beta)
    current_c_ext = compute_externality_total(candidate_state, params, cfg)

    rows = []
    for idx in range(len(params)):
        cf_state = candidate_state.copy()
        cf_state[idx] = STATE_F
        cf_c_ext = compute_externality_total(cf_state, params, cfg)

        pre_action = choose_action(
            current=int(candidate_state[idx]),
            u_n=float(utility_block["U_N_rr"][idx]),
            u_i=float(utility_block["U_E_rr"][idx]),
            u_f=float(utility_block["U_F_rr"][idx]),
        )
        if pre_action != STATE_F or cf_c_ext <= c_star or current_c_ext > c_star:
            pivot = 0.0
        else:
            noise = rng.normal(0.0, cfg.safety_fee_noise_sigma)
            pivot = float(np.clip(1.0 - noise, 0.05, 1.0))

        rr_full_sharing_gain = float(
            utility_block["U_F_rr"][idx]
            - max(float(utility_block["U_E_rr"][idx]), float(utility_block["U_N_rr"][idx]))
        )
        d_min = np.nan if pivot <= 0.0 else max(rr_full_sharing_gain / pivot, 0.0)

        rows.append(
            {
                "agent_id": int(idx),
                "current_c_ext": current_c_ext,
                "counterfactual_c_ext": cf_c_ext,
                "pivot_probability": pivot,
                "delta_P": float(params.loc[idx, "delta_P"]),
                "delta_C": float(utility_block["delta_C"][idx]),
                "rr_upgrade_gain": rr_full_sharing_gain,
                "D_min": d_min,
            }
        )
    return pd.DataFrame(rows)


def scan_safety_fee_deterrence(
    candidate_state: np.ndarray,
    params: pd.DataFrame,
    cfg: ExperimentConfig,
    c_star: float,
    beta: float,
) -> Dict[str, pd.DataFrame]:
    base_utils = compute_utilities_with_rr(candidate_state, params, cfg, beta=beta)
    current_c_ext = compute_externality_total(candidate_state, params, cfg)
    pre_state = candidate_state.copy()
    batch_upgrade_agents = []
    for idx in range(len(params)):
        pre_action = choose_action(
            current=int(candidate_state[idx]),
            u_n=float(base_utils["U_N_rr"][idx]),
            u_i=float(base_utils["U_E_rr"][idx]),
            u_f=float(base_utils["U_F_rr"][idx]),
        )
        pre_state[idx] = pre_action
        if pre_action == STATE_F:
            batch_upgrade_agents.append(idx)

    pre_c_ext = compute_externality_total(pre_state, params, cfg)
    batch_pivot = float(pre_c_ext > c_star)
    batch_upgrade_set = set(batch_upgrade_agents)

    agent_rows = []
    for idx in range(len(params)):
        cf_state = candidate_state.copy()
        cf_state[idx] = STATE_F
        cf_c_ext = compute_externality_total(cf_state, params, cfg)
        pivot = float(batch_pivot > 0.0 and idx in batch_upgrade_set)
        delta_p = float(params.loc[idx, "delta_P"])
        delta_c_i = float(base_utils["delta_C"][idx])
        gain_from_deviation = float(
            base_utils["U_F_rr"][idx]
            - max(float(base_utils["U_E_rr"][idx]), float(base_utils["U_N_rr"][idx]))
        )
        d_min = np.nan if pivot <= 0.0 else max(gain_from_deviation / pivot, 0.0)
        agent_rows.append(
            {
                "agent_id": int(idx),
                "current_c_ext": current_c_ext,
                "counterfactual_c_ext": cf_c_ext,
                "batch_counterfactual_c_ext": pre_c_ext,
                "pivot_probability": pivot,
                "delta_P": delta_p,
                "delta_C": delta_c_i,
                "gain_from_deviation": gain_from_deviation,
                "rr_upgrade_gain": gain_from_deviation,
                "D_min": d_min,
            }
        )
    agent_df = pd.DataFrame(agent_rows)

    scan_rows = []
    valid_agents = agent_df.dropna(subset=["D_min"]).copy()
    if not valid_agents.empty:
        for mult in cfg.safety_fee_multipliers:
            deterrence_flags = []
            for row in valid_agents.itertuples(index=False):
                safety_fee = mult * float(row.D_min)
                net_gain = float(row.gain_from_deviation) - float(row.pivot_probability) * safety_fee
                deterrence_flags.append(net_gain <= 1e-10)
            scan_rows.append(
                {
                    "safety_fee_multiple": float(mult),
                    "deterrence_success_rate": float(np.mean(deterrence_flags)),
                    "n_valid_agents": int(len(deterrence_flags)),
                }
            )
    return {
        "agent_df": agent_df,
        "scan_df": pd.DataFrame(scan_rows),
    }


def run_theory_closure_from_seed(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    seed_share_f: float,
    rng: np.random.Generator,
    welfare_c_star: float | None = None,
) -> Dict[str, object]:
    state = build_initial_state(cfg, seed_share_f=seed_share_f, rng=rng)
    trajectory_rows = [
        {
            "t": 0,
            "share_N": float(np.mean(state == 0)),
            "share_E": float(np.mean(state == STATE_E)),
            "share_F": float(np.mean(state == STATE_F)),
            "W": compute_social_welfare(state, params, cfg, c_star=welfare_c_star),
        }
    ]

    for t in range(1, cfg.max_rounds + 1):
        utility_block = compute_utilities(state, params, cfg)
        next_state = state.copy()
        for idx in range(cfg.n_agents):
            next_state[idx] = choose_action(
                current=int(state[idx]),
                u_n=float(utility_block["U_N"][idx]),
                u_i=float(utility_block["U_E"][idx]),
                u_f=float(utility_block["U_F"][idx]),
            )
        trajectory_rows.append(
            {
                "t": t,
                "share_N": float(np.mean(next_state == 0)),
                "share_E": float(np.mean(next_state == STATE_E)),
                "share_F": float(np.mean(next_state == STATE_F)),
                "W": compute_social_welfare(next_state, params, cfg, c_star=welfare_c_star),
            }
        )
        if np.array_equal(next_state, state):
            break
        state = next_state

    final_share_f = float(np.mean(state == STATE_F))
    max_t = trajectory_rows[-1]["t"]
    final_row = trajectory_rows[-1].copy()
    for t in range(max_t + 1, cfg.max_rounds + 1):
        padded_row = final_row.copy()
        padded_row["t"] = t
        trajectory_rows.append(padded_row)
    return {
        "final_state": state,
        "final_share_F": final_share_f,
        "high_externality_indicator": int(final_share_f >= cfg.high_state_threshold),
        "trajectory": pd.DataFrame(trajectory_rows),
    }


def run_theory_closure_with_safety_fee(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    seed_share_f: float,
    beta: float,
    safety_fee_vector: np.ndarray,
    c_star: float,
    rng: np.random.Generator,
    welfare_c_star: float | None = None,
    safety_fee_multiple: float = 1.0,
    dynamic_safety_fee: bool | None = None,
) -> Dict[str, object]:
    state = build_initial_state(cfg, seed_share_f=seed_share_f, rng=rng)
    trajectory_rows = [
        {
            "t": 0,
            "share_N": float(np.mean(state == 0)),
            "share_E": float(np.mean(state == STATE_E)),
            "share_F": float(np.mean(state == STATE_F)),
            "W": compute_social_welfare(
                state,
                params,
                cfg,
                c_star=c_star if welfare_c_star is None else welfare_c_star,
            ),
        }
    ]

    for t in range(1, cfg.max_rounds + 1):
        utility_block = compute_utilities_with_rr(
            state,
            params,
            cfg,
            beta=beta,
        )
        update_agents = np.arange(cfg.n_agents, dtype=int)
        next_state, _ = apply_batch_safety_threshold_update(
            state,
            update_agents,
            utility_block,
            params,
            cfg,
            safety_fee_vector,
            c_star,
            dynamic_safety_fee=cfg.safety_fee_dynamic if dynamic_safety_fee is None else dynamic_safety_fee,
            safety_fee_multiple=safety_fee_multiple,
        )
        trajectory_rows.append(
            {
                "t": t,
                "share_N": float(np.mean(next_state == 0)),
                "share_E": float(np.mean(next_state == STATE_E)),
                "share_F": float(np.mean(next_state == STATE_F)),
                "W": compute_social_welfare(
                    next_state,
                    params,
                    cfg,
                    c_star=c_star if welfare_c_star is None else welfare_c_star,
                ),
            }
        )
        if np.array_equal(next_state, state):
            break
        state = next_state

    final_share_f = float(np.mean(state == STATE_F))
    max_t = trajectory_rows[-1]["t"]
    final_row = trajectory_rows[-1].copy()
    for t in range(max_t + 1, cfg.max_rounds + 1):
        padded_row = final_row.copy()
        padded_row["t"] = t
        trajectory_rows.append(padded_row)
    return {
        "final_state": state,
        "final_share_F": final_share_f,
        "high_externality_indicator": int(final_share_f >= cfg.high_state_threshold),
        "trajectory": pd.DataFrame(trajectory_rows),
    }


def run_async_theory_with_safety_fee(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    beta: float,
    safety_fee_vector: np.ndarray,
    c_star: float,
    initial_state: np.ndarray,
    update_schedule: List[np.ndarray],
    welfare_c_star: float | None = None,
    safety_fee_multiple: float = 1.0,
    dynamic_safety_fee: bool | None = None,
) -> Dict[str, object]:
    state = initial_state.copy()
    trajectory_rows = []

    for t in range(cfg.max_rounds + 1):
        trajectory_rows.append(
            {
                "t": t,
                "share_N": float(np.mean(state == STATE_N)),
                "share_E": float(np.mean(state == STATE_E)),
                "share_F": float(np.mean(state == STATE_F)),
                "W": compute_social_welfare(
                    state,
                    params,
                    cfg,
                    c_star=c_star if welfare_c_star is None else welfare_c_star,
                ),
            }
        )
        if t == cfg.max_rounds or t >= len(update_schedule):
            break

        utility_block = compute_utilities_with_rr(
            state,
            params,
            cfg,
            beta=beta,
        )
        update_agents = np.asarray(update_schedule[t], dtype=int)
        state, _ = apply_batch_safety_threshold_update(
            state,
            update_agents,
            utility_block,
            params,
            cfg,
            safety_fee_vector,
            c_star,
            dynamic_safety_fee=cfg.safety_fee_dynamic if dynamic_safety_fee is None else dynamic_safety_fee,
            safety_fee_multiple=safety_fee_multiple,
        )

    return {
        "final_state": state.copy(),
        "final_share_F": float(np.mean(state == STATE_F)),
        "trajectory": pd.DataFrame(trajectory_rows),
    }


def run_theory_closure_with_rr(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    seed_share_f: float,
    beta: float,
    rng: np.random.Generator,
    welfare_c_star: float | None = None,
) -> Dict[str, object]:
    state = build_initial_state(cfg, seed_share_f=seed_share_f, rng=rng)
    trajectory_rows = [
        {
            "t": 0,
            "share_N": float(np.mean(state == STATE_N)),
            "share_E": float(np.mean(state == STATE_E)),
            "share_F": float(np.mean(state == STATE_F)),
            "W": compute_social_welfare(state, params, cfg, c_star=welfare_c_star),
        }
    ]

    for t in range(1, cfg.max_rounds + 1):
        utility_block = compute_utilities_with_rr(
            state,
            params,
            cfg,
            beta=beta,
        )
        next_state = state.copy()
        for idx in range(cfg.n_agents):
            next_state[idx] = choose_action(
                current=int(state[idx]),
                u_n=float(utility_block["U_N_rr"][idx]),
                u_i=float(utility_block["U_E_rr"][idx]),
                u_f=float(utility_block["U_F_rr"][idx]),
            )
        trajectory_rows.append(
            {
                "t": t,
                "share_N": float(np.mean(next_state == STATE_N)),
                "share_E": float(np.mean(next_state == STATE_E)),
                "share_F": float(np.mean(next_state == STATE_F)),
                "W": compute_social_welfare(next_state, params, cfg, c_star=welfare_c_star),
            }
        )
        if np.array_equal(next_state, state):
            break
        state = next_state

    final_share_f = float(np.mean(state == STATE_F))
    max_t = trajectory_rows[-1]["t"]
    final_row = trajectory_rows[-1].copy()
    for t in range(max_t + 1, cfg.max_rounds + 1):
        padded_row = final_row.copy()
        padded_row["t"] = t
        trajectory_rows.append(padded_row)
    return {
        "final_state": state,
        "final_share_F": final_share_f,
        "high_externality_indicator": int(final_share_f >= cfg.high_state_threshold),
        "trajectory": pd.DataFrame(trajectory_rows),
    }


def run_async_theory_with_rr(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    beta: float,
    initial_state: np.ndarray,
    update_schedule: List[np.ndarray],
    welfare_c_star: float | None = None,
) -> Dict[str, object]:
    state = initial_state.copy()
    trajectory_rows = []

    for t in range(cfg.max_rounds + 1):
        trajectory_rows.append(
            {
                "t": t,
                "share_N": float(np.mean(state == STATE_N)),
                "share_E": float(np.mean(state == STATE_E)),
                "share_F": float(np.mean(state == STATE_F)),
                "W": compute_social_welfare(state, params, cfg, c_star=welfare_c_star),
            }
        )
        if t == cfg.max_rounds or t >= len(update_schedule):
            break

        utility_block = compute_utilities_with_rr(
            state,
            params,
            cfg,
            beta=beta,
        )
        next_state = state.copy()
        for idx in np.asarray(update_schedule[t], dtype=int):
            next_state[idx] = choose_action(
                current=int(state[idx]),
                u_n=float(utility_block["U_N_rr"][idx]),
                u_i=float(utility_block["U_E_rr"][idx]),
                u_f=float(utility_block["U_F_rr"][idx]),
            )
        state = next_state

    return {
        "final_state": state.copy(),
        "final_share_F": float(np.mean(state == STATE_F)),
        "trajectory": pd.DataFrame(trajectory_rows),
    }
