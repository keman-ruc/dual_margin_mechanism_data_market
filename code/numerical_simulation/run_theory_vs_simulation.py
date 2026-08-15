from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sim_config import DEFAULT_OUTPUT_DIR, ExperimentConfig, config_for_profile
from sim_model import STATE_F, STATE_E, build_initial_state, config_to_dict, sample_agent_parameters
from sim_runner import run_single_simulation, run_single_simulation_with_safety_fee, run_single_simulation_with_rr
from sim_theory import (
    build_candidate_state_from_share,
    compute_entry_margin,
    compute_high_externality_candidate,
    compute_rr_theory,
    compute_social_benchmark_table,
    compute_upgrade_margin,
    scan_safety_fee_deterrence,
    scan_rr_feasibility,
    scan_rr_theory_curve,
    run_async_theory_with_safety_fee,
    run_async_theory_with_rr,
    run_theory_closure_from_seed,
    run_theory_closure_with_safety_fee,
    run_theory_closure_with_rr,
)


ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = ROOT / ".tmp_mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".tmp_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_NAMES = {
    "overview_four_props": "fig00_four_propositions_summary.png",
    "overview_three_mechanisms_share": "fig08a_three_mechanisms_share_comparison.png",
    "primitive_reference_share": "fig06a_primitive_reference_strategy_share_theory_vs_simulation.png",
    "primitive_reference_traj": "fig06b_primitive_reference_strategy_trajectory_theory_vs_simulation.png",
    "primitive_reference_f_dist": "fig06c_primitive_reference_final_share_f_distribution.png",
    "prop1_threshold": "fig03_proposition_1_theoretical_thresholds_and_first_simulated_crossings.png",
    "prop1_threshold_x15": "fig03_proposition_1_theoretical_thresholds_and_first_simulated_crossings_x15.png",
    "prop2_premium": "fig03a_private_vs_social_required_premium.png",
    "prop2_excessive": "fig04_proposition_2_private_social_and_excessive_full_sharing.png",
    "prop2_loss": "fig11_excessive_upgrade_welfare_loss.png",
    "prop3_cost_decline": "fig05_proposition_3_diminishing_marginal_privacy_cost.png",
    "prop4_beta0": "fig06_proposition_4_beta0_boundary_and_RR_feasibility.png",
    "prop4_share": "fig07_proposition_4_final_strategy_shares_theory_vs_simulation.png",
    "prop4_traj": "fig06d_prop4_rr_strategy_trajectory_theory_vs_simulation.png",
    "mech_threshold": "fig10_safety_threshold_local_deterrence_by_fee_multiple.png",
    "mech_seed": "fig13_high_externality_probability_by_initial_full_sharing_seed_share.png",
    "mech_safety_fee_seed": "fig10c_safety_seed_sensitivity.png",
    "mech_safety_fee_share": "fig11_main_mechanism_final_strategy_shares.png",
    "mech_safety_fee_traj": "fig08b_mechanism_strategy_trajectory_theory_vs_simulation.png",
    "mech_welfare_raw": "fig09_welfare_trajectory.png",
    "mech_welfare_stable": "fig09b_welfare_trajectory_stable_positive.png",
    "mech_welfare_subset": "fig09c_welfare_subset_theory_vs_simulation.png",
    "mech_welfare_t20": "fig09_welfare_trajectory_with_t20_no_theory.png",
}

FIGURE_NAMES_ALT = {}


def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)


def plot_binned_trend(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    raw_color: str,
    trend_color: str,
    raw_label: str,
    trend_label: str,
    n_bins: int = 6,
) -> None:
    if df.empty:
        return

    ax.scatter(df[x_col], df[y_col], s=22, alpha=0.22, color=raw_color, label=raw_label)

    unique_x = int(df[x_col].nunique())
    if unique_x < 2:
        return

    bin_count = min(n_bins, unique_x)
    binned = df[[x_col, y_col]].copy()
    binned["_bin"] = pd.qcut(binned[x_col], q=bin_count, duplicates="drop")
    grouped = (
        binned.groupby("_bin", observed=True, as_index=False)
        .agg(
            x_mean=(x_col, "mean"),
            y_mean=(y_col, "mean"),
            y_std=(y_col, "std"),
            n=(y_col, "size"),
        )
        .sort_values("x_mean")
    )
    if grouped.empty:
        return

    y_err = grouped["y_std"].fillna(0.0).to_numpy() / np.sqrt(grouped["n"].clip(lower=1).to_numpy())
    ax.errorbar(
        grouped["x_mean"],
        grouped["y_mean"],
        yerr=y_err,
        fmt="o-",
        color=trend_color,
        linewidth=1.8,
        markersize=4.5,
        capsize=3,
        label=trend_label,
    )


def plot_threshold_alignment_raw(
    ax,
    threshold_df: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if threshold_df.empty:
        ax.text(0.5, 0.5, "No threshold-crossing events observed", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = threshold_df["theoretical_threshold"].to_numpy()
    y = threshold_df["S_minus_i_cross"].to_numpy()
    rng = np.random.default_rng(20260519)
    x_span = float(max(np.max(x) - np.min(x), 1e-6))
    y_span = float(max(np.max(y) - np.min(y), 1e-6))
    x = x + rng.normal(0.0, 0.006 * x_span, size=x.shape)
    y = y + rng.normal(0.0, 0.010 * y_span, size=y.shape)
    x = np.clip(x, 0.0, None)
    y = np.clip(y, 0.0, None)
    bound = float(max(np.max(x), np.max(y), 1e-6) * 1.02)
    ax.scatter(x, y, s=18, alpha=0.42, color="#4C72B0", label="First crossing events")
    ax.plot([0, bound], [0, bound], linestyle="--", color="#C44E52", linewidth=1.8, label="Theory 45-degree line")
    ax.set_xlim(0, bound)
    ax.set_ylim(0, bound)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")


def theory_tables_for_run(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    final_state: np.ndarray,
    beta_rr: float,
) -> Dict[str, object]:
    entry_df = pd.DataFrame(
        {
            "agent_id": params["agent_id"],
            "theoretical_entry_margin": compute_entry_margin(params),
            "observed_final_action": final_state,
        }
    )

    upgrade_df = pd.DataFrame(
        {
            "agent_id": params["agent_id"],
            "theoretical_threshold": params["theoretical_threshold"],
            "observed_final_action": final_state,
            "observed_upgrade_margin": compute_upgrade_margin(final_state, params, cfg),
        }
    )

    high_state = compute_high_externality_candidate(params, cfg)
    high_state_df = pd.DataFrame(
        {
            "agent_id": params["agent_id"],
            "in_compromise_group": params["agent_id"].isin(high_state["compromise_idx"]).astype(int),
            "S_minus_i_h": high_state["S_minus_i_h"],
            "delta_C_h": high_state["delta_C_h"],
            "delta_P": params["delta_P"],
            "U_N_h": high_state["U_N_h"],
            "U_F_h": high_state["U_F_h"],
            "local_equilibrium_condition": high_state["local_eq_mask"],
        }
    )

    rr_state = high_state["state_h"]
    rr_params = params.copy()
    rr_params["pi_solo"] = (
        rr_params["pi_solo"] * cfg.rr_feasibility_solo_multiplier
    )
    rr_params["pi_break"] = rr_params["pi_solo"]
    rr = compute_rr_theory(rr_state, rr_params, cfg)
    rr_scan = scan_rr_feasibility(rr_state, rr_params, cfg)

    candidate_meta_rows: List[Dict[str, object]] = []
    candidate_objects: List[Dict[str, object]] = []
    for share in cfg.safety_fee_candidate_share_grid:
        candidate = build_candidate_state_from_share(params, cfg, share)
        social_sim_df = compute_social_benchmark_table(candidate["state"], params, cfg)
        candidate_row = {
            "candidate_share_f": float(candidate["share_f"]),
            "candidate_c_ext": float(candidate["c_ext"]),
            "candidate_local_eq_rate": float(candidate["local_eq_rate"]),
            "private_upgrade_rate": float(social_sim_df["private_upgrade_indicator"].mean()),
            "social_upgrade_rate": float(social_sim_df["social_upgrade_indicator"].mean()),
            "excessive_upgrade_rate": float(social_sim_df["excessive_upgrade_indicator"].mean()),
            "observed_f_share": float(social_sim_df["observed_f_indicator"].mean()),
            "observed_excessive_f_share": float(social_sim_df["observed_excessive_f_indicator"].mean()),
        }
        candidate_meta_rows.append(candidate_row)
        candidate_objects.append(
            {
                **candidate_row,
                "state": candidate["state"],
            }
        )

    candidate_meta_df = pd.DataFrame(candidate_meta_rows)
    if candidate_meta_df.empty:
        candidate_meta_df = pd.DataFrame(
            [
                {
                    "candidate_share_f": float(cfg.safety_fee_candidate_share_grid[0]),
                    "candidate_c_ext": 0.0,
                    "candidate_local_eq_rate": 0.0,
                    "private_upgrade_rate": 0.0,
                    "social_upgrade_rate": 0.0,
                    "excessive_upgrade_rate": 0.0,
                    "observed_f_share": 0.0,
                    "observed_excessive_f_share": 0.0,
                }
            ]
        )
        candidate_objects = [
            {
                **candidate_meta_df.iloc[0].to_dict(),
                "state": build_candidate_state_from_share(params, cfg, cfg.safety_fee_candidate_share_grid[0])["state"],
            }
        ]

    eligible_candidates = [
        item
        for item in candidate_objects
        if item["candidate_local_eq_rate"] > 0.0 and item["candidate_share_f"] >= 0.16
    ]
    if not eligible_candidates:
        eligible_candidates = [item for item in candidate_objects if item["candidate_local_eq_rate"] > 0.0]
    ranking_pool = eligible_candidates if eligible_candidates else candidate_objects
    best_candidate = max(
        ranking_pool,
        key=lambda item: (
            float(item["candidate_share_f"]),
            float(item["excessive_upgrade_rate"]),
            float(item["candidate_local_eq_rate"]),
            float(item["private_upgrade_rate"] - item["social_upgrade_rate"]),
            -abs(float(item["candidate_share_f"]) - 0.24),
        ),
    )

    c_star = float(best_candidate["candidate_c_ext"])
    safety_fee = scan_safety_fee_deterrence(best_candidate["state"], params, cfg, c_star=c_star, beta=beta_rr)
    safety_fee_agent_df = safety_fee["agent_df"].copy()
    if not safety_fee_agent_df.empty:
        safety_fee_agent_df["candidate_share_f"] = best_candidate["candidate_share_f"]
        safety_fee_agent_df["candidate_local_eq_rate"] = best_candidate["candidate_local_eq_rate"]
        safety_fee_agent_df["c_star"] = c_star

    safety_fee_scan_df = safety_fee["scan_df"].copy()
    if not safety_fee_scan_df.empty:
        safety_fee_scan_df["candidate_share_f"] = best_candidate["candidate_share_f"]
        safety_fee_scan_df["candidate_local_eq_rate"] = best_candidate["candidate_local_eq_rate"]
        safety_fee_scan_df["c_star"] = c_star

    selected_candidate_meta_df = pd.DataFrame(
        [
            {
                "candidate_share_f": best_candidate["candidate_share_f"],
                "candidate_c_ext": c_star,
                "candidate_local_eq_rate": best_candidate["candidate_local_eq_rate"],
                "private_upgrade_rate": best_candidate["private_upgrade_rate"],
                "social_upgrade_rate": best_candidate["social_upgrade_rate"],
                "excessive_upgrade_rate": best_candidate["excessive_upgrade_rate"],
                "observed_f_share": best_candidate["observed_f_share"],
                "observed_excessive_f_share": best_candidate["observed_excessive_f_share"],
            }
        ]
    )

    return {
        "entry": entry_df,
        "upgrade": upgrade_df,
        "high_state": high_state_df,
        "rr_beta0": rr["beta0_df"],
        "rr_scan": rr_scan,
        "safety_fee_theory": safety_fee_agent_df,
        "safety_fee_scan": safety_fee_scan_df,
        "candidate_meta": candidate_meta_df,
        "selected_candidate_meta": selected_candidate_meta_df,
    }


def build_safety_fee_vector_from_candidate(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    candidate_state: np.ndarray,
    c_star: float,
    beta: float,
) -> np.ndarray:
    safety_fee = scan_safety_fee_deterrence(candidate_state, params, cfg, c_star=c_star, beta=beta)
    safety_fee_vector = np.zeros(cfg.n_agents, dtype=float)
    agent_df = safety_fee["agent_df"].dropna(subset=["D_min"]).copy()
    for row in agent_df.itertuples(index=False):
        safety_fee_vector[int(row.agent_id)] = float(row.D_min)
    return safety_fee_vector


def select_rr_beta(cfg: ExperimentConfig, params: pd.DataFrame) -> float:
    high_state = compute_high_externality_candidate(params, cfg)
    rr_scan = scan_rr_feasibility(high_state["state_h"], params, cfg)
    feasible_beta = rr_scan[rr_scan["all_core_feasible"] >= 1].sort_values("beta")
    if not feasible_beta.empty:
        return float(feasible_beta.iloc[0]["beta"])
    return float(rr_scan.sort_values("beta").iloc[0]["beta"])


def build_shared_update_schedule(
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    update_count = max(1, int(round(cfg.n_agents * cfg.async_share)))
    return [
        rng.choice(cfg.n_agents, size=update_count, replace=False).astype(int)
        for _ in range(cfg.max_rounds)
    ]


def monte_carlo_experiment(cfg: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    run_rows: List[Dict[str, object]] = []
    crossing_rows: List[Dict[str, object]] = []
    switch_rows: List[Dict[str, object]] = []
    seed_rows: List[Dict[str, object]] = []
    seed_theory_rows: List[Dict[str, object]] = []
    rr_seed_rows: List[Dict[str, object]] = []
    rr_seed_theory_rows: List[Dict[str, object]] = []
    safety_fee_seed_rows: List[Dict[str, object]] = []
    safety_fee_seed_theory_rows: List[Dict[str, object]] = []
    strategy_share_rows: List[Dict[str, object]] = []
    strategy_trajectory_rows: List[Dict[str, object]] = []
    rr_strategy_share_rows: List[Dict[str, object]] = []
    rr_strategy_trajectory_rows: List[Dict[str, object]] = []
    safety_fee_strategy_share_rows: List[Dict[str, object]] = []
    safety_fee_strategy_trajectory_rows: List[Dict[str, object]] = []
    welfare_trajectory_rows: List[Dict[str, object]] = []
    welfare_subset_rows: List[Dict[str, object]] = []
    rr_beta_rows: List[Dict[str, object]] = []
    rr_scan_rows: List[Dict[str, object]] = []
    rr_theory_curve_rows: List[Dict[str, object]] = []
    safety_fee_rows: List[Dict[str, object]] = []
    safety_fee_scan_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    social_benchmark_rows: List[Dict[str, object]] = []
    social_benchmark_summary_rows: List[Dict[str, object]] = []
    high_state_scores: List[float] = []

    for run_id in range(cfg.n_runs):
        run_seed = cfg.seed + run_id
        rng = np.random.default_rng(run_seed)
        params = sample_agent_parameters(cfg, rng)

        seed_share = cfg.low_seed_share if run_id < cfg.n_runs // 2 else cfg.high_seed_share
        shared_initial_state = build_initial_state(
            cfg,
            seed_share_f=seed_share,
            rng=np.random.default_rng(run_seed + 1001),
        )
        shared_update_schedule = build_shared_update_schedule(
            cfg,
            np.random.default_rng(run_seed + 1002),
        )
        result = run_single_simulation(
            cfg,
            params,
            seed_share_f=seed_share,
            rng=np.random.default_rng(run_seed + 1),
            initial_state=shared_initial_state,
            update_schedule=shared_update_schedule,
        )
        summary = result["summary"]
        final_state = result["final_state"]
        beta_star = select_rr_beta(cfg, params)

        run_rows.append(
            {
                "run_id": run_id,
                "seed_share_f": seed_share,
                **summary,
            }
        )
        strategy_share_rows.append(
            {
                "run_id": run_id,
                "source": "simulation",
                "share_N": float(summary["final_share_N"]),
                "share_E": float(summary["final_share_E"]),
                "share_F": float(summary["final_share_F"]),
            }
        )
        sim_traj = result["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
        sim_traj["run_id"] = run_id
        sim_traj["source"] = "simulation"
        strategy_trajectory_rows.extend(sim_traj.to_dict(orient="records"))
        sim_welfare_traj = result["trajectory"][["t", "W"]].copy()
        sim_welfare_traj["run_id"] = run_id
        sim_welfare_traj["source"] = "simulation"
        sim_welfare_traj["mechanism"] = "primitive_reference"
        welfare_trajectory_rows.extend(sim_welfare_traj.to_dict(orient="records"))
        sim_local_welfare = result.get("local_welfare", pd.DataFrame())
        if not sim_local_welfare.empty:
            sim_local_welfare = sim_local_welfare.copy()
            sim_local_welfare["run_id"] = run_id
            sim_local_welfare["source"] = "simulation"
            sim_local_welfare["mechanism"] = "primitive_reference"
            welfare_subset_rows.extend(sim_local_welfare.to_dict(orient="records"))

        if not result["crossing_events"].empty:
            cross_df = result["crossing_events"].copy()
            cross_df = cross_df[cross_df["t_cross"] > 0].copy()
            if not cross_df.empty:
                cross_df["run_id"] = run_id
                crossing_rows.extend(cross_df.to_dict(orient="records"))

        if not result["switch_events"].empty:
            switch_df = result["switch_events"].copy()
            switch_df["run_id"] = run_id
            switch_rows.extend(switch_df.to_dict(orient="records"))

        theory = theory_tables_for_run(cfg, params, final_state, beta_rr=beta_star)
        high_state_scores.append(float(theory["high_state"]["local_equilibrium_condition"].mean()))

        theory_closure = run_theory_closure_from_seed(
            cfg,
            params,
            seed_share_f=seed_share,
            rng=np.random.default_rng(run_seed + 3),
        )
        theory_state = theory_closure["final_state"]
        strategy_share_rows.append(
            {
                "run_id": run_id,
                "source": "theory_closure",
                "share_N": float(np.mean(theory_state == 0)),
                "share_E": float(np.mean(theory_state == STATE_E)),
                "share_F": float(np.mean(theory_state == STATE_F)),
            }
        )
        theory_traj = theory_closure["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
        theory_traj["run_id"] = run_id
        theory_traj["source"] = "theory_closure"
        strategy_trajectory_rows.extend(theory_traj.to_dict(orient="records"))
        theory_welfare_traj = theory_closure["trajectory"][["t", "W"]].copy()
        theory_welfare_traj["run_id"] = run_id
        theory_welfare_traj["source"] = "theory_closure"
        theory_welfare_traj["mechanism"] = "primitive_reference"
        welfare_trajectory_rows.extend(theory_welfare_traj.to_dict(orient="records"))

        candidate_meta_df = theory["candidate_meta"].copy()
        candidate_meta_df["run_id"] = run_id
        candidate_meta_df["source"] = "candidate_state"
        social_benchmark_summary_rows.extend(candidate_meta_df.to_dict(orient="records"))

        for candidate_row in candidate_meta_df.to_dict(orient="records"):
            candidate_state_tmp = build_candidate_state_from_share(params, cfg, float(candidate_row["candidate_share_f"]))["state"]
            candidate_social_df = compute_social_benchmark_table(candidate_state_tmp, params, cfg)
            candidate_social_df["run_id"] = run_id
            candidate_social_df["source"] = "candidate_state"
            candidate_social_df["candidate_share_f"] = float(candidate_row["candidate_share_f"])
            social_benchmark_rows.extend(candidate_social_df.to_dict(orient="records"))

        selected_candidate = theory["selected_candidate_meta"].iloc[0]
        candidate_state = build_candidate_state_from_share(params, cfg, float(selected_candidate["candidate_share_f"]))["state"]
        c_star = float(selected_candidate["candidate_c_ext"])

        safety_fee_vector = build_safety_fee_vector_from_candidate(
            cfg,
            params,
            candidate_state,
            c_star=c_star,
            beta=beta_star,
        )

        safety_fee_result = run_single_simulation_with_safety_fee(
            cfg,
            params,
            seed_share_f=seed_share,
            beta=beta_star,
            safety_fee_vector=safety_fee_vector,
            c_star=c_star,
            rng=np.random.default_rng(run_seed + 4),
            initial_state=shared_initial_state,
            update_schedule=shared_update_schedule,
            welfare_c_star=c_star,
        )
        dep_summary = safety_fee_result["summary"]
        safety_fee_strategy_share_rows.append(
            {
                "run_id": run_id,
                "source": "simulation_safety_fee",
                "share_N": float(dep_summary["final_share_N"]),
                "share_E": float(dep_summary["final_share_E"]),
                "share_F": float(dep_summary["final_share_F"]),
                "beta_rr": beta_star,
            }
        )
        dep_traj = safety_fee_result["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
        dep_traj["run_id"] = run_id
        dep_traj["source"] = "simulation_safety_fee"
        safety_fee_strategy_trajectory_rows.extend(dep_traj.to_dict(orient="records"))
        dep_welfare_traj = safety_fee_result["trajectory"][["t", "W"]].copy()
        dep_welfare_traj["run_id"] = run_id
        dep_welfare_traj["source"] = "simulation_safety_fee"
        dep_welfare_traj["mechanism"] = "safety_fee"
        welfare_trajectory_rows.extend(dep_welfare_traj.to_dict(orient="records"))
        dep_local_welfare = safety_fee_result.get("local_welfare", pd.DataFrame())
        if not dep_local_welfare.empty:
            dep_local_welfare = dep_local_welfare.copy()
            dep_local_welfare["run_id"] = run_id
            dep_local_welfare["source"] = "simulation_safety_fee"
            dep_local_welfare["mechanism"] = "safety_fee"
            welfare_subset_rows.extend(dep_local_welfare.to_dict(orient="records"))

        safety_fee_theory = run_theory_closure_with_safety_fee(
            cfg,
            params,
            seed_share_f=seed_share,
            beta=beta_star,
            safety_fee_vector=safety_fee_vector,
            c_star=c_star,
            rng=np.random.default_rng(run_seed + 5),
            welfare_c_star=c_star,
        )
        dep_theory_state = safety_fee_theory["final_state"]
        safety_fee_strategy_share_rows.append(
            {
                "run_id": run_id,
                "source": "theory_safety_fee",
                "share_N": float(np.mean(dep_theory_state == 0)),
                "share_E": float(np.mean(dep_theory_state == STATE_E)),
                "share_F": float(np.mean(dep_theory_state == STATE_F)),
                "beta_rr": beta_star,
            }
        )
        dep_theory_traj = safety_fee_theory["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
        dep_theory_traj["run_id"] = run_id
        dep_theory_traj["source"] = "theory_safety_fee"
        safety_fee_strategy_trajectory_rows.extend(dep_theory_traj.to_dict(orient="records"))
        dep_theory_welfare_traj = safety_fee_theory["trajectory"][["t", "W"]].copy()
        dep_theory_welfare_traj["run_id"] = run_id
        dep_theory_welfare_traj["source"] = "theory_safety_fee"
        dep_theory_welfare_traj["mechanism"] = "safety_fee"
        welfare_trajectory_rows.extend(dep_theory_welfare_traj.to_dict(orient="records"))

        dep_async_theory = run_async_theory_with_safety_fee(
            cfg,
            params,
            beta=beta_star,
            safety_fee_vector=safety_fee_vector,
            c_star=c_star,
            initial_state=shared_initial_state,
            update_schedule=shared_update_schedule,
            welfare_c_star=c_star,
        )
        dep_async_theory_state = dep_async_theory["final_state"]
        safety_fee_strategy_share_rows.append(
            {
                "run_id": run_id,
                "source": "theory_safety_fee_async",
                "share_N": float(np.mean(dep_async_theory_state == 0)),
                "share_E": float(np.mean(dep_async_theory_state == STATE_E)),
                "share_F": float(np.mean(dep_async_theory_state == STATE_F)),
                "beta_rr": beta_star,
            }
        )
        dep_async_theory_traj = dep_async_theory["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
        dep_async_theory_traj["run_id"] = run_id
        dep_async_theory_traj["source"] = "theory_safety_fee_async"
        safety_fee_strategy_trajectory_rows.extend(dep_async_theory_traj.to_dict(orient="records"))
        dep_async_theory_welfare_traj = dep_async_theory["trajectory"][["t", "W"]].copy()
        dep_async_theory_welfare_traj["run_id"] = run_id
        dep_async_theory_welfare_traj["source"] = "theory_safety_fee_async"
        dep_async_theory_welfare_traj["mechanism"] = "safety_fee"
        welfare_trajectory_rows.extend(dep_async_theory_welfare_traj.to_dict(orient="records"))

        rr_tmp = theory["rr_beta0"].copy()
        if not rr_tmp.empty:
            rr_tmp["run_id"] = run_id
            rr_beta_rows.extend(rr_tmp.to_dict(orient="records"))
            theory_curve_tmp = scan_rr_theory_curve(theory["rr_beta0"], cfg)
            if not theory_curve_tmp.empty:
                theory_curve_tmp["run_id"] = run_id
                rr_theory_curve_rows.extend(theory_curve_tmp.to_dict(orient="records"))

        rr_scan_tmp = theory["rr_scan"].copy()
        if not rr_scan_tmp.empty:
            rr_scan_tmp["run_id"] = run_id
            rr_scan_rows.extend(rr_scan_tmp.to_dict(orient="records"))
            rr_result = run_single_simulation_with_rr(
                cfg,
                params,
                seed_share_f=seed_share,
                beta=beta_star,
                rng=np.random.default_rng(run_seed + 6),
                initial_state=shared_initial_state,
                update_schedule=shared_update_schedule,
                welfare_c_star=c_star,
            )
            rho_summary = rr_result["summary"]
            rr_strategy_share_rows.append(
                {
                    "run_id": run_id,
                    "source": "simulation_rr",
                    "share_N": float(rho_summary["final_share_N"]),
                    "share_E": float(rho_summary["final_share_E"]),
                    "share_F": float(rho_summary["final_share_F"]),
                    "beta_rr": beta_star,
                }
            )
            rho_traj = rr_result["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
            rho_traj["run_id"] = run_id
            rho_traj["source"] = "simulation_rr"
            rho_traj["beta_rr"] = beta_star
            rr_strategy_trajectory_rows.extend(rho_traj.to_dict(orient="records"))
            rho_welfare_traj = rr_result["trajectory"][["t", "W"]].copy()
            rho_welfare_traj["run_id"] = run_id
            rho_welfare_traj["source"] = "simulation_rr"
            rho_welfare_traj["beta_rr"] = beta_star
            rho_welfare_traj["mechanism"] = "rr"
            welfare_trajectory_rows.extend(rho_welfare_traj.to_dict(orient="records"))
            rho_local_welfare = rr_result.get("local_welfare", pd.DataFrame())
            if not rho_local_welfare.empty:
                rho_local_welfare = rho_local_welfare.copy()
                rho_local_welfare["run_id"] = run_id
                rho_local_welfare["source"] = "simulation_rr"
                rho_local_welfare["beta_rr"] = beta_star
                rho_local_welfare["mechanism"] = "rr"
                welfare_subset_rows.extend(rho_local_welfare.to_dict(orient="records"))

            rr_theory = run_theory_closure_with_rr(
                cfg,
                params,
                seed_share_f=seed_share,
                beta=beta_star,
                rng=np.random.default_rng(run_seed + 7),
                welfare_c_star=c_star,
            )
            rho_theory_state = rr_theory["final_state"]
            rr_strategy_share_rows.append(
                {
                    "run_id": run_id,
                    "source": "theory_rr",
                    "share_N": float(np.mean(rho_theory_state == 0)),
                    "share_E": float(np.mean(rho_theory_state == STATE_E)),
                    "share_F": float(np.mean(rho_theory_state == STATE_F)),
                    "beta_rr": beta_star,
                }
            )
            rho_theory_traj = rr_theory["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
            rho_theory_traj["run_id"] = run_id
            rho_theory_traj["source"] = "theory_rr"
            rho_theory_traj["beta_rr"] = beta_star
            rr_strategy_trajectory_rows.extend(rho_theory_traj.to_dict(orient="records"))
            rho_theory_welfare_traj = rr_theory["trajectory"][["t", "W"]].copy()
            rho_theory_welfare_traj["run_id"] = run_id
            rho_theory_welfare_traj["source"] = "theory_rr"
            rho_theory_welfare_traj["beta_rr"] = beta_star
            rho_theory_welfare_traj["mechanism"] = "rr"
            welfare_trajectory_rows.extend(rho_theory_welfare_traj.to_dict(orient="records"))

            rr_async_theory = run_async_theory_with_rr(
                cfg,
                params,
                beta=beta_star,
                initial_state=shared_initial_state,
                update_schedule=shared_update_schedule,
                welfare_c_star=c_star,
            )
            rho_async_theory_state = rr_async_theory["final_state"]
            rr_strategy_share_rows.append(
                {
                    "run_id": run_id,
                    "source": "theory_rr_async",
                    "share_N": float(np.mean(rho_async_theory_state == 0)),
                    "share_E": float(np.mean(rho_async_theory_state == STATE_E)),
                    "share_F": float(np.mean(rho_async_theory_state == STATE_F)),
                    "beta_rr": beta_star,
                }
            )
            rho_async_theory_traj = rr_async_theory["trajectory"][["t", "share_N", "share_E", "share_F"]].copy()
            rho_async_theory_traj["run_id"] = run_id
            rho_async_theory_traj["source"] = "theory_rr_async"
            rho_async_theory_traj["beta_rr"] = beta_star
            rr_strategy_trajectory_rows.extend(rho_async_theory_traj.to_dict(orient="records"))
            rho_async_theory_welfare_traj = rr_async_theory["trajectory"][["t", "W"]].copy()
            rho_async_theory_welfare_traj["run_id"] = run_id
            rho_async_theory_welfare_traj["source"] = "theory_rr_async"
            rho_async_theory_welfare_traj["beta_rr"] = beta_star
            rho_async_theory_welfare_traj["mechanism"] = "rr"
            welfare_trajectory_rows.extend(rho_async_theory_welfare_traj.to_dict(orient="records"))

        dep_tmp = theory["safety_fee_theory"].copy()
        if not dep_tmp.empty:
            dep_tmp["run_id"] = run_id
            safety_fee_rows.extend(dep_tmp.to_dict(orient="records"))

        dep_scan_tmp = theory["safety_fee_scan"].copy()
        if not dep_scan_tmp.empty:
            dep_scan_tmp["run_id"] = run_id
            safety_fee_scan_rows.extend(dep_scan_tmp.to_dict(orient="records"))

        candidate_tmp = theory["candidate_meta"].copy()
        candidate_tmp["run_id"] = run_id
        candidate_rows.extend(candidate_tmp.to_dict(orient="records"))

    for seed_share in cfg.seed_scan_grid:
        success_flags = []
        mean_final_share_f = []
        theory_high_flags = []
        theory_final_share_f = []
        rr_success_flags = []
        rr_mean_final_share_f = []
        rr_theory_high_flags = []
        rr_theory_final_share_f = []
        safety_fee_success_flags = []
        safety_fee_mean_final_share_f = []
        safety_fee_theory_high_flags = []
        safety_fee_theory_final_share_f = []
        for rep in range(max(8, cfg.n_runs // 3)):
            rep_seed = cfg.seed + 1000 + int(seed_share * 1000) * 10 + rep
            rng = np.random.default_rng(rep_seed)
            params = sample_agent_parameters(cfg, rng)
            result = run_single_simulation(cfg, params, seed_share_f=seed_share, rng=np.random.default_rng(rep_seed + 1))
            success_flags.append(int(result["summary"]["steady_state_label"] == "high_externality"))
            mean_final_share_f.append(float(result["summary"]["final_share_F"]))
            theory_result = run_theory_closure_from_seed(
                cfg,
                params,
                seed_share_f=seed_share,
                rng=np.random.default_rng(rep_seed + 2),
            )
            theory_high_flags.append(int(theory_result["high_externality_indicator"]))
            theory_final_share_f.append(float(theory_result["final_share_F"]))

            high_state = compute_high_externality_candidate(params, cfg)
            rr_scan_tmp = scan_rr_feasibility(high_state["state_h"], params, cfg)
            beta_star = select_rr_beta(cfg, params)

            rr_result = run_single_simulation_with_rr(
                cfg,
                params,
                seed_share_f=seed_share,
                beta=beta_star,
                rng=np.random.default_rng(rep_seed + 3),
            )
            rr_success_flags.append(int(rr_result["summary"]["steady_state_label"] == "high_externality"))
            rr_mean_final_share_f.append(float(rr_result["summary"]["final_share_F"]))
            rr_theory_result = run_theory_closure_with_rr(
                cfg,
                params,
                seed_share_f=seed_share,
                beta=beta_star,
                rng=np.random.default_rng(rep_seed + 4),
            )
            rr_theory_high_flags.append(int(rr_theory_result["high_externality_indicator"]))
            rr_theory_final_share_f.append(float(rr_theory_result["final_share_F"]))

            best_candidate = None
            for share in cfg.safety_fee_candidate_share_grid:
                candidate = build_candidate_state_from_share(params, cfg, share)
                if candidate["local_eq_rate"] > 0.0:
                    if best_candidate is None or candidate["local_eq_rate"] > best_candidate["local_eq_rate"]:
                        best_candidate = candidate
            if best_candidate is None:
                best_candidate = build_candidate_state_from_share(params, cfg, cfg.safety_fee_candidate_share_grid[0])
            c_star = float(best_candidate["c_ext"])
            safety_fee_vector = build_safety_fee_vector_from_candidate(
                cfg,
                params,
                best_candidate["state"],
                c_star=c_star,
                beta=beta_star,
            )

            safety_fee_result = run_single_simulation_with_safety_fee(
                cfg,
                params,
                seed_share_f=seed_share,
                beta=beta_star,
                safety_fee_vector=safety_fee_vector,
                c_star=c_star,
                rng=np.random.default_rng(rep_seed + 5),
                welfare_c_star=c_star,
            )
            safety_fee_success_flags.append(int(safety_fee_result["summary"]["steady_state_label"] == "high_externality"))
            safety_fee_mean_final_share_f.append(float(safety_fee_result["summary"]["final_share_F"]))
            safety_fee_theory_result = run_theory_closure_with_safety_fee(
                cfg,
                params,
                seed_share_f=seed_share,
                beta=beta_star,
                safety_fee_vector=safety_fee_vector,
                c_star=c_star,
                rng=np.random.default_rng(rep_seed + 6),
                welfare_c_star=c_star,
            )
            safety_fee_theory_high_flags.append(int(safety_fee_theory_result["high_externality_indicator"]))
            safety_fee_theory_final_share_f.append(float(safety_fee_theory_result["final_share_F"]))
        seed_rows.append(
            {
                "seed_share_f": seed_share,
                "high_externality_probability": float(np.mean(success_flags)),
                "mean_final_share_F": float(np.mean(mean_final_share_f)),
            }
        )
        seed_theory_rows.append(
            {
                "seed_share_f": seed_share,
                "theory_high_externality_rate": float(np.mean(theory_high_flags)),
                "theory_mean_final_share_F": float(np.mean(theory_final_share_f)),
            }
        )
        rr_seed_rows.append(
            {
                "seed_share_f": seed_share,
                "high_externality_probability": float(np.mean(rr_success_flags)),
                "mean_final_share_F": float(np.mean(rr_mean_final_share_f)),
            }
        )
        rr_seed_theory_rows.append(
            {
                "seed_share_f": seed_share,
                "theory_high_externality_rate": float(np.mean(rr_theory_high_flags)),
                "theory_mean_final_share_F": float(np.mean(rr_theory_final_share_f)),
            }
        )
        safety_fee_seed_rows.append(
            {
                "seed_share_f": seed_share,
                "high_externality_probability": float(np.mean(safety_fee_success_flags)),
                "mean_final_share_F": float(np.mean(safety_fee_mean_final_share_f)),
            }
        )
        safety_fee_seed_theory_rows.append(
            {
                "seed_share_f": seed_share,
                "theory_high_externality_rate": float(np.mean(safety_fee_theory_high_flags)),
                "theory_mean_final_share_F": float(np.mean(safety_fee_theory_final_share_f)),
            }
        )

    combined_share_rows = []
    mechanism_specs = [
        ("primitive_reference", "simulation", "theory_closure", strategy_share_rows),
        ("rr", "simulation_rr", "theory_rr_async", rr_strategy_share_rows),
        ("safety_threshold_dm", "simulation_safety_fee", "theory_safety_fee_async", safety_fee_strategy_share_rows),
    ]
    for mechanism, sim_source, theory_source, rows in mechanism_specs:
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        for source_label, source_key in [("simulation", sim_source), ("theory", theory_source)]:
            subset = df[df["source"] == source_key]
            if subset.empty:
                continue
            means = subset[["share_N", "share_E", "share_F"]].mean()
            combined_share_rows.append(
                {
                    "mechanism": mechanism,
                    "source": source_label,
                    "share_N": float(means["share_N"]),
                    "share_E": float(means["share_E"]),
                    "share_F": float(means["share_F"]),
                }
            )

    return {
        "run_summary": pd.DataFrame(run_rows),
        "threshold_events": pd.DataFrame(crossing_rows),
        "switch_events": pd.DataFrame(switch_rows),
        "seed_scan": pd.DataFrame(seed_rows),
        "seed_theory_scan": pd.DataFrame(seed_theory_rows),
        "rr_seed_scan": pd.DataFrame(rr_seed_rows),
        "rr_seed_theory_scan": pd.DataFrame(rr_seed_theory_rows),
        "safety_fee_seed_scan": pd.DataFrame(safety_fee_seed_rows),
        "safety_fee_seed_theory_scan": pd.DataFrame(safety_fee_seed_theory_rows),
        "strategy_share_comparison": pd.DataFrame(strategy_share_rows),
        "strategy_trajectory_comparison": pd.DataFrame(strategy_trajectory_rows),
        "rr_strategy_share_comparison": pd.DataFrame(rr_strategy_share_rows),
        "rr_strategy_trajectory_comparison": pd.DataFrame(rr_strategy_trajectory_rows),
        "safety_fee_strategy_share_comparison": pd.DataFrame(safety_fee_strategy_share_rows),
        "safety_fee_strategy_trajectory_comparison": pd.DataFrame(safety_fee_strategy_trajectory_rows),
        "welfare_trajectory_comparison": pd.DataFrame(welfare_trajectory_rows),
        "welfare_subset_comparison": pd.DataFrame(welfare_subset_rows),
        "three_mechanisms_share_comparison": pd.DataFrame(combined_share_rows),
        "rr_beta0": pd.DataFrame(rr_beta_rows),
        "rr_scan": pd.DataFrame(rr_scan_rows),
        "rr_theory_curve": pd.DataFrame(rr_theory_curve_rows),
        "safety_fee_theory": pd.DataFrame(safety_fee_rows),
        "safety_fee_scan": pd.DataFrame(safety_fee_scan_rows),
        "candidate_state_meta": pd.DataFrame(candidate_rows),
        "social_benchmark_agent_comparison": pd.DataFrame(social_benchmark_rows),
        "social_benchmark_summary_comparison": pd.DataFrame(social_benchmark_summary_rows),
        "high_state_summary": pd.DataFrame(
            [{"mean_local_equilibrium_rate": float(np.mean(high_state_scores))}]
        ),
    }


def plot_outputs(cfg: ExperimentConfig, data: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    figures = output_dir / "figures"
    run_df = data["run_summary"]
    threshold_df = data["threshold_events"]
    switch_df = data["switch_events"]
    seed_df = data["seed_scan"]
    seed_theory_df = data["seed_theory_scan"]
    rr_seed_df = data["rr_seed_scan"]
    rr_seed_theory_df = data["rr_seed_theory_scan"]
    safety_fee_seed_df = data["safety_fee_seed_scan"]
    safety_fee_seed_theory_df = data["safety_fee_seed_theory_scan"]
    safety_only_seed_df = data.get("safety_only_seed_scan", pd.DataFrame())
    strategy_share_df = data["strategy_share_comparison"]
    strategy_trajectory_df = data["strategy_trajectory_comparison"]
    rr_strategy_share_df = data["rr_strategy_share_comparison"]
    rr_strategy_trajectory_df = data["rr_strategy_trajectory_comparison"]
    safety_fee_strategy_share_df = data["safety_fee_strategy_share_comparison"]
    safety_fee_strategy_trajectory_df = data["safety_fee_strategy_trajectory_comparison"]
    welfare_trajectory_df = data["welfare_trajectory_comparison"]
    welfare_subset_df = data["welfare_subset_comparison"]
    three_mechanisms_share_df = data["three_mechanisms_share_comparison"]
    rr_df = data["rr_beta0"]
    rr_scan_df = data["rr_scan"]
    rr_theory_curve_df = data["rr_theory_curve"]
    safety_fee_scan_df = data["safety_fee_scan"]
    social_benchmark_agent_df = data["social_benchmark_agent_comparison"]
    social_benchmark_summary_df = data["social_benchmark_summary_comparison"]

    def padded_welfare_for_plot(df: pd.DataFrame, sources: List[str]) -> pd.DataFrame:
        """Extend early-converged runs with their terminal welfare for plotting."""
        if df.empty:
            return df.copy()
        max_t = int(max(cfg.max_rounds, df["t"].max()))
        rows: List[pd.DataFrame] = []
        for source in sources:
            source_df = df[df["source"] == source]
            if source_df.empty:
                continue
            for run_id, run_df in source_df.groupby("run_id"):
                indexed = run_df.sort_values("t").set_index("t")
                padded = indexed.reindex(range(max_t + 1)).ffill()
                padded["t"] = padded.index
                padded["source"] = source
                padded["run_id"] = run_id
                rows.append(padded.reset_index(drop=True))
        if not rows:
            return df[df["source"].isin(sources)].copy()
        return pd.concat(rows, ignore_index=True)

    def padded_strategy_for_plot(df: pd.DataFrame, sources: List[str]) -> pd.DataFrame:
        """Extend early-converged strategy paths with their terminal shares."""
        if df.empty:
            return df.copy()
        max_t = int(max(cfg.max_rounds, df["t"].max()))
        untouched = df[~df["source"].isin(sources)].copy()
        rows: List[pd.DataFrame] = [untouched] if not untouched.empty else []
        for source in sources:
            source_df = df[df["source"] == source]
            for run_id, run_df in source_df.groupby("run_id"):
                indexed = run_df.sort_values("t").set_index("t")
                padded = indexed.reindex(range(max_t + 1)).ffill()
                padded["t"] = padded.index
                padded["source"] = source
                padded["run_id"] = run_id
                rows.append(padded.reset_index(drop=True))
        return pd.concat(rows, ignore_index=True) if rows else df.copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_threshold_alignment_raw(
        ax,
        threshold_df,
        title="Proposition 1(a)-(c): threshold vs first crossing",
        xlabel="Theoretical threshold S_bar",
        ylabel="First simulated crossing S_-i",
    )
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop1_threshold"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_threshold_alignment_raw(
        ax,
        threshold_df,
        title="Proposition 1(a)-(c): threshold vs first crossing",
        xlabel="Theoretical threshold S_bar",
        ylabel="First simulated crossing S_-i",
    )
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop1_threshold_x15"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if threshold_df.empty:
        ax.text(0.5, 0.5, "No threshold-crossing events observed", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped_cross = threshold_df.sort_values(["run_id", "t_cross", "agent_id"]).copy()
        plot_binned_trend(
            ax,
            grouped_cross,
            "S_minus_i_cross",
            "delta_C_at_cross",
            raw_color="#4C72B0",
            trend_color="#C44E52",
            raw_label="Crossing observations",
            trend_label="Binned mean",
        )
        ax.set_xlabel("Crossing environment S_-i")
        ax.set_ylabel("Incremental privacy cost Delta C_i")
        ax.set_title("Proposition 3: marginal privacy cost decline")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop3_cost_decline"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(
        safety_fee_seed_df["seed_share_f"],
        safety_fee_seed_df["high_externality_probability"],
        where="post",
        linewidth=2.2,
        label="Simulated probability",
    )
    if not safety_fee_seed_theory_df.empty:
        ax.step(
            safety_fee_seed_theory_df["seed_share_f"],
            safety_fee_seed_theory_df["theory_high_externality_rate"],
            where="post",
            linewidth=2.0,
            color="#C44E52",
            label="Theory closure prediction",
        )
    ax.set_xlabel("Initial F seed share")
    ax.set_ylabel("Probability of high-externality steady state")
    ax.set_title("Safety-threshold (DM) basin-of-attraction scan")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_safety_fee_seed"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        rr_seed_df["seed_share_f"],
        rr_seed_df["high_externality_probability"],
        marker="o",
        linewidth=2.2,
        label="RR",
    )
    ax.plot(
        safety_fee_seed_df["seed_share_f"],
        safety_fee_seed_df["high_externality_probability"],
        marker="o",
        linewidth=2.2,
        label="DM",
    )
    if not safety_only_seed_df.empty:
        ax.plot(
            safety_only_seed_df["seed_share_f"],
            safety_only_seed_df["high_externality_probability"],
            marker="o",
            linewidth=2.2,
            label="ST",
        )
    ax.plot(
        seed_df["seed_share_f"],
        seed_df["high_externality_probability"],
        marker="o",
        linewidth=2.2,
        label="PM",
    )
    ax.set_xlabel("Initial F seed share")
    ax.set_ylabel("Probability of high-externality steady state")
    ax.set_title("Basin-of-attraction comparison across mechanisms")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_seed"], dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    if rr_scan_df.empty:
        ax1.text(0.5, 0.5, "No RR scan results observed", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()
    else:
        grouped_scan = rr_scan_df.groupby("beta", as_index=False)["all_core_feasible"].mean()
        ax1.plot(grouped_scan["beta"], grouped_scan["all_core_feasible"], marker="o", linewidth=2.2, color="#4C72B0", label="Simulated feasible rate")
        if not rr_theory_curve_df.empty:
            grouped_theory_curve = rr_theory_curve_df.groupby("beta", as_index=False)["all_core_feasible_theory"].mean()
            ax1.plot(
                grouped_theory_curve["beta"],
                grouped_theory_curve["all_core_feasible_theory"],
                linestyle="--",
                linewidth=2.0,
                color="#C44E52",
                label="Theory feasibility curve",
            )
        ax1.set_xlabel("RR rate beta")
        ax1.set_ylabel("Run-level feasibility rate")
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(alpha=0.3)

        ax1.legend(loc="lower left")
        ax1.set_title("Proposition 4: stressed beta_0 vs RR feasibility")
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop4_beta0"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if safety_fee_scan_df.empty:
        ax.text(0.5, 0.5, "No positive-pivot agents in candidate safe states", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped_dep = safety_fee_scan_df.groupby("safety_fee_multiple", as_index=False)["deterrence_success_rate"].mean()
        ax.step(
            grouped_dep["safety_fee_multiple"],
            grouped_dep["deterrence_success_rate"],
            where="post",
            linewidth=2.2,
            color="#1f77b4",
            label="Simulated deterrence success",
        )
        ax.scatter(
            grouped_dep["safety_fee_multiple"],
            grouped_dep["deterrence_success_rate"],
            s=52,
            color="#1f77b4",
            zorder=3,
        )
        ax.axvline(1.0, linestyle="--", color="#C44E52", linewidth=1.5, label="Theory threshold D / D_min = 1")
        ax.step(
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 1.0],
            where="post",
            linestyle=":",
            linewidth=2.0,
            color="#8172B3",
            label="Theory deterrence rule",
        )
        ax.set_xlabel("Safety-threshold fee multiple D / D_min")
        ax.set_ylabel("Local deterrence success rate")
        ax.set_title("Safety-threshold (DM) local-threshold comparison")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_threshold"], dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    if social_benchmark_agent_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No social benchmark comparison available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        subset = social_benchmark_agent_df[social_benchmark_agent_df["source"] == "candidate_state"]
        if subset.empty:
            subset = social_benchmark_agent_df
        bins = [
            ("Low candidate F share", subset["candidate_share_f"] <= 0.12),
            ("Middle candidate F share", (subset["candidate_share_f"] > 0.12) & (subset["candidate_share_f"] <= 0.24)),
            ("High candidate F share", subset["candidate_share_f"] > 0.24),
        ]
        for ax, (title, mask) in zip(axes, bins):
            part = subset.loc[mask]
            if part.empty:
                ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            x = part["private_required_premium"].to_numpy()
            y = part["social_required_premium"].to_numpy()
            bound = max(float(np.max(x)), float(np.max(y)), 1e-6) * 1.05
            ax.scatter(x, y, s=16, alpha=0.25, color="#4C72B0")
            ax.plot([0, bound], [0, bound], linestyle="--", color="#C44E52", linewidth=1.6)
            ax.set_xlim(0, bound)
            ax.set_ylim(0, bound)
            share_min = float(part["candidate_share_f"].min())
            share_max = float(part["candidate_share_f"].max())
            ax.set_title(f"{title}\n(candidate F share {share_min:.2f}-{share_max:.2f})")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Social required premium")
        for ax in axes:
            ax.set_xlabel("Private required premium")
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color="#4C72B0", alpha=0.45, markersize=5, label="Run-agent observations"),
            plt.Line2D([0], [0], linestyle="--", color="#C44E52", linewidth=1.6, label="45-degree line"),
        ]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.98))
        fig.suptitle("Proposition 2(a): private vs social upgrade requirement", y=1.03)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop2_premium"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if social_benchmark_summary_df.empty:
        ax.text(0.5, 0.5, "No excessive full-sharing summary available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped = social_benchmark_summary_df.groupby("source", as_index=False)[
            ["private_upgrade_rate", "social_upgrade_rate", "excessive_upgrade_rate"]
        ].mean()
        grouped = grouped.set_index("source").reindex(["candidate_state"]).fillna(0.0)
        labels = ["Private upgrade", "Social upgrade", "Excessive upgrade"]
        x = np.arange(len(labels))
        vals = grouped.loc["candidate_state", ["private_upgrade_rate", "social_upgrade_rate", "excessive_upgrade_rate"]].to_numpy()
        ax.bar(x, vals, width=0.55, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Average share")
        ax.set_title("Proposition 2(b): socially excessive full sharing")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop2_excessive"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if social_benchmark_agent_df.empty:
        ax.text(0.5, 0.5, "No Prop. 2 welfare-loss data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        subset = social_benchmark_agent_df[social_benchmark_agent_df["source"] == "candidate_state"].copy()
        if subset.empty:
            subset = social_benchmark_agent_df.copy()
        subset["welfare_loss_if_excessive"] = np.where(
            subset["excessive_upgrade_indicator"] == 1,
            np.maximum(-subset["social_margin"], 0.0),
            0.0,
        )
        run_curve = (
            subset.groupby(["run_id", "candidate_share_f"], as_index=False)
            .agg(
                excessive_share=("excessive_upgrade_indicator", "mean"),
                total_welfare_loss=("welfare_loss_if_excessive", "sum"),
                avg_loss_per_agent=("welfare_loss_if_excessive", "mean"),
            )
        )
        curve = (
            run_curve.groupby("candidate_share_f", as_index=False)
            .agg(
                total_welfare_loss_mean=("total_welfare_loss", "mean"),
                total_welfare_loss_std=("total_welfare_loss", "std"),
                excessive_share_mean=("excessive_share", "mean"),
                avg_loss_per_agent_mean=("avg_loss_per_agent", "mean"),
            )
            .sort_values("candidate_share_f")
        )
        if curve.empty:
            ax.text(0.5, 0.5, "No excessive-upgrade observations", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            x = curve["candidate_share_f"].to_numpy()
            y = curve["total_welfare_loss_mean"].to_numpy()
            yerr = curve["total_welfare_loss_std"].fillna(0.0).to_numpy()
            ax.plot(
                x,
                y,
                color="#C44E52",
                linewidth=2.4,
                marker="o",
                markersize=5.5,
                label="Aggregate welfare loss from excessive upgrades",
            )
            ax.fill_between(
                x,
                np.maximum(y - yerr, 0.0),
                y + yerr,
                color="#C44E52",
                alpha=0.18,
                label="Across-run dispersion",
            )
            ax.set_xlabel("Candidate F share")
            ax.set_ylabel("Average aggregate welfare loss")
            ax.set_title("Proposition 2(c): social welfare loss from excessive upgrades")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop2_loss"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(run_df["final_share_F"], bins=18, alpha=0.85, color="#55A868")
    ax.axvline(cfg.high_state_threshold, linestyle="--", color="#C44E52", linewidth=1.6, label="Theory high-state threshold")
    ax.axvline(0.10, linestyle=":", color="#4C72B0", linewidth=1.6, label="Theory low-state threshold")
    ax.set_xlabel("Final F share")
    ax.set_ylabel("Count")
    ax.set_title("Primitive reference dynamic outcome distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["primitive_reference_f_dist"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if strategy_share_df.empty:
        ax.text(0.5, 0.5, "No strategy share comparison available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped = strategy_share_df.groupby("source", as_index=False)[["share_N", "share_E", "share_F"]].mean()
        sim_row = grouped[grouped["source"] == "simulation"]
        theory_row = grouped[grouped["source"] == "theory_closure"]
        labels = ["N", "E", "F"]
        x = np.arange(len(labels))
        width = 0.34
        sim_vals = sim_row[["share_N", "share_E", "share_F"]].to_numpy().flatten() if not sim_row.empty else np.zeros(3)
        theory_vals = theory_row[["share_N", "share_E", "share_F"]].to_numpy().flatten() if not theory_row.empty else np.zeros(3)
        ax.bar(x - width / 2, theory_vals, width=width, color="#C44E52", alpha=0.85, label="Theory closure")
        ax.bar(x + width / 2, sim_vals, width=width, color="#4C72B0", alpha=0.85, label="Simulation mean")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Average final share")
        ax.set_title("Final N / E / F share: theory vs simulation")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["primitive_reference_share"], dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    if strategy_trajectory_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No trajectory comparison available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        grouped_traj = (
            strategy_trajectory_df.groupby(["source", "t"], as_index=False)[["share_N", "share_E", "share_F"]]
            .mean()
        )
        styles = {
            "theory_closure": ("--", "#C44E52", "Theory closure"),
            "simulation": ("-", "#4C72B0", "Simulation mean"),
        }
        series_info = [("share_N", "N share"), ("share_E", "E share"), ("share_F", "F share")]
        for ax, (col, title) in zip(axes, series_info):
            for source, (linestyle, color, label) in styles.items():
                subset = grouped_traj[grouped_traj["source"] == source]
                if not subset.empty:
                    ax.plot(subset["t"], subset[col], linestyle=linestyle, linewidth=2.0, color=color, label=label)
            ax.set_ylabel(title)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)
        axes[0].set_title("N / E / F trajectory: theory vs simulation")
        axes[-1].set_xlabel("Round t")
        axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["primitive_reference_traj"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if rr_strategy_share_df.empty:
        ax.text(0.5, 0.5, "No RR strategy share comparison available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped = rr_strategy_share_df.groupby("source", as_index=False)[["share_N", "share_E", "share_F"]].mean()
        subset = grouped.set_index("source").reindex(["theory_rr_async", "simulation_rr"]).fillna(0.0)
        labels = ["Theory", "Simulation"]
        colors = {"share_N": "#4C72B0", "share_E": "#55A868", "share_F": "#C44E52"}
        x = np.arange(len(labels))
        width = 0.24
        for offset, (col, legend_label) in enumerate([("share_N", "N"), ("share_E", "E"), ("share_F", "F")]):
            vals = subset[col].to_numpy()
            bars = ax.bar(
                x + (offset - 1) * width,
                vals,
                width=width,
                color=colors[col],
                alpha=0.9,
                label=legend_label,
            )
            for bar, val in zip(bars, vals):
                if val >= 0.035:
                    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Strategy share")
        ax.set_title("RR-regime final strategy shares")
        ax.legend(ncols=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop4_share"], dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    if rr_strategy_trajectory_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No RR trajectory comparison available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        rr_plot_df = padded_strategy_for_plot(rr_strategy_trajectory_df, ["simulation_rr"])
        grouped_traj = (
            rr_plot_df.groupby(["source", "t"], as_index=False)[["share_N", "share_E", "share_F"]]
            .mean()
        )
        styles = {
            "theory_rr_async": ("--", "#C44E52", "Async theory"),
            "simulation_rr": ("-", "#4C72B0", "Simulation mean"),
        }
        series_info = [("share_N", "N share"), ("share_E", "E share"), ("share_F", "F share")]
        for ax, (col, title) in zip(axes, series_info):
            for source, (linestyle, color, label) in styles.items():
                subset = grouped_traj[grouped_traj["source"] == source]
                if not subset.empty:
                    ax.plot(subset["t"], subset[col], linestyle=linestyle, linewidth=2.0, color=color, label=label)
            ax.set_ylabel(title)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)
        axes[0].set_title("RR N / E / F trajectory: theory vs simulation")
        axes[-1].set_xlabel("Round t")
        axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["prop4_traj"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if safety_fee_strategy_share_df.empty:
        ax.text(0.5, 0.5, "No safety-threshold strategy share comparison available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped = safety_fee_strategy_share_df.groupby("source", as_index=False)[["share_E", "share_F"]].mean()
        sim_row = grouped[grouped["source"] == "simulation_safety_fee"]
        theory_row = grouped[grouped["source"] == "theory_safety_fee_async"]
        labels = ["E", "F"]
        x = np.arange(len(labels))
        width = 0.34
        sim_vals = sim_row[["share_E", "share_F"]].to_numpy().flatten() if not sim_row.empty else np.zeros(2)
        theory_vals = theory_row[["share_E", "share_F"]].to_numpy().flatten() if not theory_row.empty else np.zeros(2)
        ax.bar(x - width / 2, theory_vals, width=width, color="#C44E52", alpha=0.85, label="Async theory")
        ax.bar(x + width / 2, sim_vals, width=width, color="#4C72B0", alpha=0.85, label="Simulation mean")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Average final share")
        ax.set_title("Safety-threshold (DM) final E / F share")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_safety_fee_share"], dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    if safety_fee_strategy_trajectory_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No safety-threshold trajectory comparison available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        safety_fee_plot_df = padded_strategy_for_plot(safety_fee_strategy_trajectory_df, ["simulation_safety_fee"])
        grouped_traj = (
            safety_fee_plot_df.groupby(["source", "t"], as_index=False)[["share_N", "share_E", "share_F"]]
            .mean()
        )
        styles = {
            "theory_safety_fee_async": ("--", "#C44E52", "Async theory"),
            "simulation_safety_fee": ("-", "#4C72B0", "Simulation mean"),
        }
        series_info = [("share_N", "N share"), ("share_E", "E share"), ("share_F", "F share")]
        for ax, (col, title) in zip(axes, series_info):
            for source, (linestyle, color, label) in styles.items():
                subset = grouped_traj[grouped_traj["source"] == source]
                if not subset.empty:
                    ax.plot(subset["t"], subset[col], linestyle=linestyle, linewidth=2.0, color=color, label=label)
            ax.set_ylabel(title)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.3)
        axes[0].set_title("Safety-threshold (DM) N / E / F trajectory")
        axes[-1].set_xlabel("Round t")
        axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_safety_fee_traj"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if three_mechanisms_share_df.empty:
        ax.text(0.5, 0.5, "No mechanism comparison available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        mechanisms = ["rr_theory", "rr_sim", "safety_fee_theory", "safety_fee_sim"]
        labels = ["RR\ntheory", "RR\nsim", "Safety\ntheory", "Safety\nsim"]
        colors = {"share_N": "#4C72B0", "share_E": "#55A868", "share_F": "#C44E52"}
        rr_grouped = (
            rr_strategy_share_df.groupby("source", as_index=False)[["share_N", "share_E", "share_F"]]
            .mean()
            .set_index("source")
        )
        safety_fee_grouped = (
            safety_fee_strategy_share_df.groupby("source", as_index=False)[["share_N", "share_E", "share_F"]]
            .mean()
            .set_index("source")
        )
        subset = pd.DataFrame(
            [
                rr_grouped.reindex(["theory_rr_async"]).fillna(0.0).iloc[0],
                rr_grouped.reindex(["simulation_rr"]).fillna(0.0).iloc[0],
                safety_fee_grouped.reindex(["theory_safety_fee_async"]).fillna(0.0).iloc[0],
                safety_fee_grouped.reindex(["simulation_safety_fee"]).fillna(0.0).iloc[0],
            ],
            index=mechanisms,
        )
        x = np.arange(len(mechanisms))
        width = 0.24
        for offset, (col, legend_label) in enumerate([("share_N", "N"), ("share_E", "E"), ("share_F", "F")]):
            vals = subset[col].to_numpy()
            bars = ax.bar(
                x + (offset - 1) * width,
                vals,
                width=width,
                color=colors[col],
                alpha=0.9,
                label=legend_label,
            )
            for bar, val in zip(bars, vals):
                if val >= 0.035:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.018,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Mean final strategy share")
        ax.set_title("Final N/E/F shares by mechanism")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncols=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["overview_three_mechanisms_share"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if welfare_trajectory_df.empty:
        ax.text(0.5, 0.5, "No welfare trajectory available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        plot_sources = ["simulation_rr", "theory_rr_async", "simulation_safety_fee", "theory_safety_fee_async"]
        padded_sim = padded_welfare_for_plot(welfare_trajectory_df, ["simulation_rr", "simulation_safety_fee"])
        theory_part = welfare_trajectory_df[welfare_trajectory_df["source"].isin(["theory_rr_async", "theory_safety_fee_async"])]
        plot_welfare_df = pd.concat([padded_sim, theory_part], ignore_index=True)
        grouped_w = plot_welfare_df.groupby(["source", "t"], as_index=False).agg(W=("W", "mean"), n=("W", "size"))
        styles = {
            "simulation_rr": ("-", "#55A868", "RR sim"),
            "theory_rr_async": ("--", "#8172B3", "RR async theory"),
            "simulation_safety_fee": ("-", "#DD8452", "Safety-threshold sim"),
            "theory_safety_fee_async": ("--", "#8C8C8C", "Safety-threshold async theory"),
        }
        for source in plot_sources:
            linestyle, color, label = styles[source]
            subset = grouped_w[grouped_w["source"] == source]
            if not subset.empty:
                ax.plot(subset["t"], subset["W"], linestyle=linestyle, linewidth=2.0, color=color, label=label)
        ax.set_xlim(0, max(int(plot_welfare_df["t"].max()), 1))
        ax.set_xlabel("Round t")
        ax.set_ylabel("Social welfare W")
        ax.set_title("Welfare trajectory comparison (padded after convergence)")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_welfare_raw"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if welfare_trajectory_df.empty:
        ax.text(0.5, 0.5, "No welfare trajectory available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        stable_cutoff = cfg.max_rounds
        plot_sources = ["simulation_rr", "theory_rr_async", "simulation_safety_fee", "theory_safety_fee_async"]
        padded_sim = padded_welfare_for_plot(welfare_trajectory_df, ["simulation_rr", "simulation_safety_fee"])
        theory_part = welfare_trajectory_df[welfare_trajectory_df["source"].isin(["theory_rr_async", "theory_safety_fee_async"])]
        plot_df = pd.concat([padded_sim, theory_part], ignore_index=True)
        plot_df = plot_df[plot_df["source"].isin(plot_sources) & (plot_df["t"] <= stable_cutoff)].copy()
        plot_grouped = plot_df.groupby(["source", "t"], as_index=False)["W"].mean()
        styles = {
            "simulation_rr": ("-", "#55A868", "RR sim"),
            "theory_rr_async": ("--", "#8172B3", "RR async theory"),
            "simulation_safety_fee": ("-", "#DD8452", "Safety-threshold sim"),
            "theory_safety_fee_async": ("--", "#8C8C8C", "Safety-threshold async theory"),
        }
        for source in plot_sources:
            linestyle, color, label = styles[source]
            subset = plot_grouped[plot_grouped["source"] == source]
            if not subset.empty:
                ax.plot(subset["t"], subset["W"], linestyle=linestyle, linewidth=2.0, color=color, label=label)
        ax.set_xlim(0, stable_cutoff)
        ax.set_xlabel("Round t")
        ax.set_ylabel("Social welfare W")
        ax.set_title("Stable-window welfare trajectory")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_welfare_stable"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if welfare_subset_df.empty:
        ax.text(0.5, 0.5, "No subset welfare trajectory available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped_ws = welfare_subset_df.groupby(["source", "t"], as_index=False).agg(
            W_subset_before=("W_subset_before", "mean"),
            W_subset_after=("W_subset_after", "mean"),
        )
        styles = {
            "simulation_rr": ("#4C72B0", "RR subset"),
            "simulation_safety_fee": ("#DD8452", "Safety-threshold subset"),
        }
        for source, (color, label) in styles.items():
            subset = grouped_ws[grouped_ws["source"] == source]
            if not subset.empty:
                ax.plot(subset["t"], subset["W_subset_before"], linestyle="--", linewidth=1.4, color=color, alpha=0.5)
                ax.plot(subset["t"], subset["W_subset_after"], linestyle="-", linewidth=2.0, color=color, label=label)
        ax.set_xlabel("Round t")
        ax.set_ylabel("Subset welfare")
        ax.set_title("Subset welfare on asynchronously updated agents")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["mech_welfare_subset"], dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    plot_threshold_alignment_raw(
        ax,
        threshold_df,
        title="Prop. 1: multi-strategy threshold",
        xlabel="Theory",
        ylabel="Simulation",
    )

    ax = axes[0, 1]
    if social_benchmark_summary_df.empty:
        ax.text(0.5, 0.5, "No Prop. 2 data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped = social_benchmark_summary_df.groupby("source", as_index=False)[
            ["private_upgrade_rate", "social_upgrade_rate", "excessive_upgrade_rate"]
        ].mean()
        grouped = grouped.set_index("source").reindex(["candidate_state"]).fillna(0.0)
        vals = grouped.loc["candidate_state", ["private_upgrade_rate", "social_upgrade_rate", "excessive_upgrade_rate"]].to_numpy()
        labels = ["Private", "Social", "Excessive"]
        ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"], alpha=0.88)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Prop. 2: welfare benchmark")
        ax.set_ylabel("Average share")
        ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    if threshold_df.empty:
        ax.text(0.5, 0.5, "No Prop. 3 data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped_cross = threshold_df.sort_values(["run_id", "t_cross", "agent_id"]).copy()
        plot_binned_trend(
            ax,
            grouped_cross,
            "S_minus_i_cross",
            "delta_C_at_cross",
            raw_color="#55A868",
            trend_color="#C44E52",
            raw_label="Crossing observations",
            trend_label="Binned mean",
            n_bins=5,
        )
        ax.set_title("Prop. 3: diminishing marginal privacy cost")
        ax.set_xlabel("Exposure environment")
        ax.set_ylabel("Observed Delta C_i")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if rr_scan_df.empty:
        ax.text(0.5, 0.5, "No Prop. 4 data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        grouped_scan = rr_scan_df.groupby("beta", as_index=False)["all_core_feasible"].mean()
        ax.plot(grouped_scan["beta"], grouped_scan["all_core_feasible"], marker="o", linewidth=2.2, color="#4C72B0", label="Simulated feasible rate")
        if not rr_theory_curve_df.empty:
            grouped_theory_curve = rr_theory_curve_df.groupby("beta", as_index=False)["all_core_feasible_theory"].mean()
            ax.plot(
                grouped_theory_curve["beta"],
                grouped_theory_curve["all_core_feasible_theory"],
                linestyle="--",
                linewidth=2.0,
                color="#C44E52",
                label="Theory feasibility curve",
            )
        ax.set_xlabel("RR rate beta")
        ax.set_ylabel("Run-level feasibility rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Prop. 4: RR feasibility boundary")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Four-Proposition Numerical Summary", y=0.98)
    fig.tight_layout()
    fig.savefig(figures / FIGURE_NAMES["overview_four_props"], dpi=220)
    plt.close(fig)


def write_outputs(cfg: ExperimentConfig, data: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    for name, df in data.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, ensure_ascii=False, indent=2)

    summary = {
        "n_runs": int(len(data["run_summary"])),
        "mean_final_share_F": float(data["run_summary"]["final_share_F"].mean()),
        "high_externality_rate": float(
            (data["run_summary"]["steady_state_label"] == "high_externality").mean()
        ),
        "mixed_state_rate": float(
            (data["run_summary"]["steady_state_label"] == "mixed_state").mean()
        ),
        "low_externality_rate": float(
            (data["run_summary"]["steady_state_label"] == "low_externality").mean()
        ),
        "mean_local_equilibrium_rate": float(
            data["high_state_summary"].iloc[0]["mean_local_equilibrium_rate"]
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run theory-vs-simulation experiments.")
    parser.add_argument("--profile", default="paper", choices=["quick", "paper"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--systemic-penalty-chi", type=float, default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from CSV files already present in --output-dir.",
    )
    return parser.parse_args()


def load_existing_outputs(output_dir: Path) -> Dict[str, pd.DataFrame]:
    data = {path.stem: pd.read_csv(path) for path in output_dir.glob("*.csv")}
    required = {
        "run_summary",
        "threshold_events",
        "switch_events",
        "seed_scan",
        "seed_theory_scan",
        "rr_seed_scan",
        "rr_seed_theory_scan",
        "safety_fee_seed_scan",
        "safety_fee_seed_theory_scan",
        "strategy_share_comparison",
        "strategy_trajectory_comparison",
        "rr_strategy_share_comparison",
        "rr_strategy_trajectory_comparison",
        "safety_fee_strategy_share_comparison",
        "safety_fee_strategy_trajectory_comparison",
        "welfare_trajectory_comparison",
        "welfare_subset_comparison",
        "three_mechanisms_share_comparison",
        "rr_beta0",
        "rr_scan",
        "rr_theory_curve",
        "safety_fee_scan",
        "social_benchmark_agent_comparison",
        "social_benchmark_summary_comparison",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise FileNotFoundError(f"missing CSV outputs for plotting: {', '.join(missing)}")
    return data


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cfg = config_for_profile(args.profile, output_dir=output_dir)
    if args.max_rounds is not None:
        cfg.max_rounds = args.max_rounds
    if args.systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = args.systemic_penalty_chi

    ensure_dirs(cfg.output_dir)
    if args.plot_only:
        config_path = cfg.output_dir / "config.json"
        if config_path.exists():
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            for key, value in saved_config.items():
                if key != "output_dir" and hasattr(cfg, key):
                    setattr(cfg, key, value)
        plot_outputs(cfg, load_existing_outputs(cfg.output_dir), cfg.output_dir)
        print(f"Regenerated figures from existing outputs in {cfg.output_dir}.")
        return

    data = monte_carlo_experiment(cfg)
    write_outputs(cfg, data, cfg.output_dir)
    plot_outputs(cfg, data, cfg.output_dir)

    print("Theory-vs-simulation experiment finished.")
    print(json.dumps(json.load(open(cfg.output_dir / "summary.json", "r", encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
