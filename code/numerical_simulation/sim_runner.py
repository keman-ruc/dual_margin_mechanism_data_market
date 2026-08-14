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
    classify_steady_state,
    compute_utilities,
    compute_utilities_with_rhood,
    compute_externality_total,
    compute_social_welfare,
    compute_social_welfare_subset,
)


def run_single_simulation(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    seed_share_f: float,
    rng: np.random.Generator,
    initial_state: np.ndarray | None = None,
    update_schedule: List[np.ndarray] | None = None,
    welfare_c_star: float | None = None,
) -> Dict[str, object]:
    state = (
        initial_state.copy()
        if initial_state is not None
        else build_initial_state(cfg, seed_share_f=seed_share_f, rng=rng)
    )
    update_count = max(1, int(round(cfg.n_agents * cfg.async_share)))

    trajectories: List[Dict[str, float]] = []
    local_welfare_rows: List[Dict[str, float]] = []
    switch_events: List[Dict[str, float]] = []
    crossing_events: List[Dict[str, float]] = []
    state_history: List[np.ndarray] = []
    share_f_history: List[float] = []
    crossed_to_f = np.zeros(cfg.n_agents, dtype=bool)
    prev_upgrade_margin: np.ndarray | None = None

    for t in range(cfg.max_rounds + 1):
        utility_block = compute_utilities(state, params, cfg)
        s_minus_i = utility_block["S_minus_i"]

        share_n = float(np.mean(state == STATE_N))
        share_i = float(np.mean(state == STATE_E))
        share_f = float(np.mean(state == STATE_F))
        c_ext = compute_externality_total(state, params, cfg)

        trajectories.append(
            {
                "t": t,
                "share_N": share_n,
                "share_E": share_i,
                "share_F": share_f,
                "C_ext": c_ext,
                "W": compute_social_welfare(state, params, cfg, c_star=welfare_c_star),
                "sensitive_mass_F": float(np.sum((state == STATE_F) * params["d_i"].to_numpy())),
            }
        )
        state_history.append(state.copy())
        share_f_history.append(share_f)

        if t == cfg.max_rounds:
            break

        if len(state_history) >= cfg.convergence_window:
            recent_states = state_history[-cfg.convergence_window :]
            if all(np.array_equal(recent_states[0], item) for item in recent_states[1:]):
                break
            recent_share_f = share_f_history[-cfg.convergence_window :]
            if max(recent_share_f) - min(recent_share_f) < cfg.convergence_eps:
                break

        next_state = state.copy()
        if update_schedule is not None and t < len(update_schedule):
            update_agents = np.asarray(update_schedule[t], dtype=int)
        else:
            update_agents = rng.choice(cfg.n_agents, size=update_count, replace=False)
        local_welfare_rows.append(
            {
                "t": int(t),
                "subset_size": int(len(update_agents)),
                "W_subset_before": compute_social_welfare_subset(
                    state, params, cfg, update_agents, c_star=welfare_c_star
                ),
            }
        )

        upgrade_margin = utility_block["U_F"] - utility_block["U_E"]
        for idx in range(cfg.n_agents):
            was_below = prev_upgrade_margin is not None and prev_upgrade_margin[idx] < -1e-10
            is_crossing = upgrade_margin[idx] >= -1e-10 and was_below
            if not crossed_to_f[idx] and is_crossing:
                crossing_events.append(
                    {
                        "agent_id": int(idx),
                        "t_cross": int(t),
                        "S_minus_i_cross": float(s_minus_i[idx]),
                        "theoretical_threshold": float(params.loc[idx, "theoretical_threshold"]),
                        "state_at_cross": int(state[idx]),
                        "upgrade_margin_at_cross": float(upgrade_margin[idx]),
                        "delta_C_at_cross": float(utility_block["delta_C"][idx]),
                        "delta_P": float(params.loc[idx, "delta_P"]),
                        "v_i": float(params.loc[idx, "v_i"]),
                        "d_i": float(params.loc[idx, "d_i"]),
                    }
                )
                crossed_to_f[idx] = True

        for idx in update_agents:
            new_action = choose_action(
                current=int(state[idx]),
                u_n=float(utility_block["U_N"][idx]),
                u_i=float(utility_block["U_E"][idx]),
                u_f=float(utility_block["U_F"][idx]),
            )
            if state[idx] == STATE_E and new_action == STATE_F:
                switch_events.append(
                    {
                        "agent_id": int(idx),
                        "t_switch": int(t + 1),
                        "S_minus_i_switch": float(s_minus_i[idx]),
                        "theoretical_threshold": float(params.loc[idx, "theoretical_threshold"]),
                        "upgrade_margin_at_switch": float(upgrade_margin[idx]),
                        "delta_C_at_switch": float(utility_block["delta_C"][idx]),
                        "delta_P": float(params.loc[idx, "delta_P"]),
                        "v_i": float(params.loc[idx, "v_i"]),
                        "d_i": float(params.loc[idx, "d_i"]),
                    }
                )
            next_state[idx] = new_action
        state = next_state
        local_welfare_rows[-1]["W_subset_after"] = compute_social_welfare_subset(
            state, params, cfg, update_agents, c_star=welfare_c_star
        )
        prev_upgrade_margin = upgrade_margin.copy()

    trajectory_df = pd.DataFrame(trajectories)
    switch_df = pd.DataFrame(switch_events)
    crossing_df = pd.DataFrame(crossing_events)
    converged = len(trajectory_df) - 1 < cfg.max_rounds
    final_share_f = float(trajectory_df.iloc[-1]["share_F"])

    summary = {
        "rounds_completed": int(trajectory_df.iloc[-1]["t"]),
        "converged": int(converged),
        "final_share_N": float(trajectory_df.iloc[-1]["share_N"]),
        "final_share_E": float(trajectory_df.iloc[-1]["share_E"]),
        "final_share_F": final_share_f,
        "final_C_ext": float(trajectory_df.iloc[-1]["C_ext"]),
        "steady_state_label": classify_steady_state(final_share_f, converged, cfg),
        "num_switch_events": int(len(switch_df)),
        "num_switch_agents": int(switch_df["agent_id"].nunique()) if not switch_df.empty else 0,
    }

    return {
        "trajectory": trajectory_df,
        "local_welfare": pd.DataFrame(local_welfare_rows),
        "switch_events": switch_df,
        "crossing_events": crossing_df,
        "summary": summary,
        "final_state": state.copy(),
    }


def run_single_simulation_with_deposit(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    seed_share_f: float,
    beta: float,
    deposit_vector: np.ndarray,
    c_star: float,
    rng: np.random.Generator,
    initial_state: np.ndarray | None = None,
    update_schedule: List[np.ndarray] | None = None,
    welfare_c_star: float | None = None,
    deposit_multiple: float = 1.0,
    dynamic_deposit: bool | None = None,
) -> Dict[str, object]:
    state = (
        initial_state.copy()
        if initial_state is not None
        else build_initial_state(cfg, seed_share_f=seed_share_f, rng=rng)
    )
    update_count = max(1, int(round(cfg.n_agents * cfg.async_share)))

    trajectories: List[Dict[str, float]] = []
    local_welfare_rows: List[Dict[str, float]] = []
    state_history: List[np.ndarray] = []
    share_f_history: List[float] = []

    for t in range(cfg.max_rounds + 1):
        utility_block = compute_utilities_with_rhood(
            state,
            params,
            cfg,
            beta=beta,
        )

        share_n = float(np.mean(state == STATE_N))
        share_i = float(np.mean(state == STATE_E))
        share_f = float(np.mean(state == STATE_F))
        c_ext = compute_externality_total(state, params, cfg)

        trajectories.append(
            {
                "t": t,
                "share_N": share_n,
                "share_E": share_i,
                "share_F": share_f,
                "C_ext": c_ext,
                "W": compute_social_welfare(
                    state,
                    params,
                    cfg,
                    c_star=c_star if welfare_c_star is None else welfare_c_star,
                ),
                "sensitive_mass_F": float(np.sum((state == STATE_F) * params["d_i"].to_numpy())),
            }
        )
        state_history.append(state.copy())
        share_f_history.append(share_f)

        if t == cfg.max_rounds:
            break

        if len(state_history) >= cfg.convergence_window:
            recent_states = state_history[-cfg.convergence_window :]
            if all(np.array_equal(recent_states[0], item) for item in recent_states[1:]):
                break
            recent_share_f = share_f_history[-cfg.convergence_window :]
            if max(recent_share_f) - min(recent_share_f) < cfg.convergence_eps:
                break

        next_state = state.copy()
        if update_schedule is not None and t < len(update_schedule):
            update_agents = np.asarray(update_schedule[t], dtype=int)
        else:
            update_agents = rng.choice(cfg.n_agents, size=update_count, replace=False)
        local_welfare_rows.append(
            {
                "t": int(t),
                "subset_size": int(len(update_agents)),
                "W_subset_before": compute_social_welfare_subset(
                    state,
                    params,
                    cfg,
                    update_agents,
                    c_star=c_star if welfare_c_star is None else welfare_c_star,
                ),
            }
        )
        next_state, batch_info = apply_batch_safety_threshold_update(
            state,
            update_agents,
            utility_block,
            params,
            cfg,
            deposit_vector,
            c_star,
            dynamic_deposit=cfg.deposit_dynamic if dynamic_deposit is None else dynamic_deposit,
            deposit_multiple=deposit_multiple,
        )
        state = next_state
        local_welfare_rows[-1]["W_subset_after"] = compute_social_welfare_subset(
            state,
            params,
            cfg,
            update_agents,
            c_star=c_star if welfare_c_star is None else welfare_c_star,
        )
        local_welfare_rows[-1].update(batch_info)

    trajectory_df = pd.DataFrame(trajectories)
    converged = len(trajectory_df) - 1 < cfg.max_rounds
    final_share_f = float(trajectory_df.iloc[-1]["share_F"])
    summary = {
        "rounds_completed": int(trajectory_df.iloc[-1]["t"]),
        "converged": int(converged),
        "final_share_N": float(trajectory_df.iloc[-1]["share_N"]),
        "final_share_E": float(trajectory_df.iloc[-1]["share_E"]),
        "final_share_F": final_share_f,
        "final_C_ext": float(trajectory_df.iloc[-1]["C_ext"]),
        "steady_state_label": classify_steady_state(final_share_f, converged, cfg),
        "beta_rhood": float(beta),
    }
    return {
        "trajectory": trajectory_df,
        "local_welfare": pd.DataFrame(local_welfare_rows),
        "summary": summary,
        "final_state": state.copy(),
    }


def run_single_simulation_with_rhood(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    seed_share_f: float,
    beta: float,
    rng: np.random.Generator,
    initial_state: np.ndarray | None = None,
    update_schedule: List[np.ndarray] | None = None,
    welfare_c_star: float | None = None,
) -> Dict[str, object]:
    state = (
        initial_state.copy()
        if initial_state is not None
        else build_initial_state(cfg, seed_share_f=seed_share_f, rng=rng)
    )
    update_count = max(1, int(round(cfg.n_agents * cfg.async_share)))

    trajectories: List[Dict[str, float]] = []
    local_welfare_rows: List[Dict[str, float]] = []
    state_history: List[np.ndarray] = []
    share_f_history: List[float] = []

    for t in range(cfg.max_rounds + 1):
        utility_block = compute_utilities_with_rhood(state, params, cfg, beta=beta)

        share_n = float(np.mean(state == STATE_N))
        share_i = float(np.mean(state == STATE_E))
        share_f = float(np.mean(state == STATE_F))
        c_ext = compute_externality_total(state, params, cfg)

        trajectories.append(
            {
                "t": t,
                "share_N": share_n,
                "share_E": share_i,
                "share_F": share_f,
                "C_ext": c_ext,
                "W": compute_social_welfare(state, params, cfg, c_star=welfare_c_star),
                "sensitive_mass_F": float(np.sum((state == STATE_F) * params["d_i"].to_numpy())),
            }
        )
        state_history.append(state.copy())
        share_f_history.append(share_f)

        if t == cfg.max_rounds:
            break

        if len(state_history) >= cfg.convergence_window:
            recent_states = state_history[-cfg.convergence_window :]
            if all(np.array_equal(recent_states[0], item) for item in recent_states[1:]):
                break
            recent_share_f = share_f_history[-cfg.convergence_window :]
            if max(recent_share_f) - min(recent_share_f) < cfg.convergence_eps:
                break

        next_state = state.copy()
        if update_schedule is not None and t < len(update_schedule):
            update_agents = np.asarray(update_schedule[t], dtype=int)
        else:
            update_agents = rng.choice(cfg.n_agents, size=update_count, replace=False)
        local_welfare_rows.append(
            {
                "t": int(t),
                "subset_size": int(len(update_agents)),
                "W_subset_before": compute_social_welfare_subset(
                    state, params, cfg, update_agents, c_star=welfare_c_star
                ),
            }
        )
        for idx in update_agents:
            new_action = choose_action(
                current=int(state[idx]),
                u_n=float(utility_block["U_N_rhood"][idx]),
                u_i=float(utility_block["U_E_rhood"][idx]),
                u_f=float(utility_block["U_F_rhood"][idx]),
            )
            next_state[idx] = new_action
        state = next_state
        local_welfare_rows[-1]["W_subset_after"] = compute_social_welfare_subset(
            state, params, cfg, update_agents, c_star=welfare_c_star
        )

    trajectory_df = pd.DataFrame(trajectories)
    converged = len(trajectory_df) - 1 < cfg.max_rounds
    final_share_f = float(trajectory_df.iloc[-1]["share_F"])
    summary = {
        "rounds_completed": int(trajectory_df.iloc[-1]["t"]),
        "converged": int(converged),
        "final_share_N": float(trajectory_df.iloc[-1]["share_N"]),
        "final_share_E": float(trajectory_df.iloc[-1]["share_E"]),
        "final_share_F": final_share_f,
        "final_C_ext": float(trajectory_df.iloc[-1]["C_ext"]),
        "steady_state_label": classify_steady_state(final_share_f, converged, cfg),
        "beta_rhood": float(beta),
    }
    return {
        "trajectory": trajectory_df,
        "local_welfare": pd.DataFrame(local_welfare_rows),
        "summary": summary,
        "final_state": state.copy(),
    }
