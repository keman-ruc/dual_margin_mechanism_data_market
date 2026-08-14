from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sim_config import config_for_profile
from sim_model import (
    STATE_F,
    STATE_E,
    build_initial_state,
    classify_steady_state,
    choose_action,
    compute_externality_total,
    compute_social_welfare,
    compute_social_welfare_subset,
    compute_utilities,
    config_to_dict,
    sample_agent_parameters,
)
from sim_runner import run_single_simulation
from sim_theory import build_candidate_state_from_share, compute_social_benchmark_table
from run_theory_vs_simulation import build_shared_update_schedule


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".tmp_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def select_safety_candidate(cfg, params: pd.DataFrame) -> Dict[str, object]:
    candidate_objects: List[Dict[str, object]] = []
    for share in cfg.deposit_candidate_share_grid:
        candidate = build_candidate_state_from_share(params, cfg, share)
        social_df = compute_social_benchmark_table(candidate["state"], params, cfg)
        candidate_objects.append(
            {
                "candidate_share_f": float(candidate["share_f"]),
                "candidate_c_ext": float(candidate["c_ext"]),
                "candidate_local_eq_rate": float(candidate["local_eq_rate"]),
                "private_upgrade_rate": float(social_df["private_upgrade_indicator"].mean()),
                "social_upgrade_rate": float(social_df["social_upgrade_indicator"].mean()),
                "excessive_upgrade_rate": float(social_df["excessive_upgrade_indicator"].mean()),
                "observed_f_share": float(social_df["observed_f_indicator"].mean()),
                "observed_excessive_f_share": float(social_df["observed_excessive_f_indicator"].mean()),
                "state": candidate["state"],
            }
        )

    eligible = [
        item
        for item in candidate_objects
        if item["candidate_local_eq_rate"] > 0.0 and item["candidate_share_f"] >= 0.16
    ]
    if not eligible:
        eligible = [item for item in candidate_objects if item["candidate_local_eq_rate"] > 0.0]
    ranking_pool = eligible if eligible else candidate_objects
    return max(
        ranking_pool,
        key=lambda item: (
            float(item["candidate_share_f"]),
            float(item["excessive_upgrade_rate"]),
            float(item["candidate_local_eq_rate"]),
            float(item["private_upgrade_rate"] - item["social_upgrade_rate"]),
            -abs(float(item["candidate_share_f"]) - 0.24),
        ),
    )


def build_primitive_safety_vector(
    candidate_state: np.ndarray,
    params: pd.DataFrame,
    cfg,
    c_star: float,
) -> np.ndarray:
    utility_block = compute_utilities(candidate_state, params, cfg)
    pre_state = candidate_state.copy()
    batch_f_agents: List[int] = []
    for idx in range(cfg.n_agents):
        pre_action = choose_action(
            current=int(candidate_state[idx]),
            u_n=float(utility_block["U_N"][idx]),
            u_i=float(utility_block["U_E"][idx]),
            u_f=float(utility_block["U_F"][idx]),
        )
        pre_state[idx] = pre_action
        if pre_action == STATE_F:
            batch_f_agents.append(idx)

    pre_c_ext = compute_externality_total(pre_state, params, cfg)
    trigger = pre_c_ext > c_star
    deposit_vector = np.zeros(cfg.n_agents, dtype=float)
    if not trigger:
        return deposit_vector

    for idx in batch_f_agents:
        gain = float(
            utility_block["U_F"][idx]
            - max(float(utility_block["U_E"][idx]), float(utility_block["U_N"][idx]))
        )
        deposit_vector[idx] = max(gain, 0.0)
    return deposit_vector


def run_single_simulation_safety_only(
    cfg,
    params: pd.DataFrame,
    seed_share_f: float,
    deposit_vector: np.ndarray,
    c_star: float,
    rng: np.random.Generator,
    initial_state: np.ndarray,
    update_schedule: List[np.ndarray],
    dynamic_deposit: bool = True,
    deposit_multiple: float = 1.0,
) -> Dict[str, object]:
    state = initial_state.copy()
    trajectories: List[Dict[str, float]] = []
    local_rows: List[Dict[str, float]] = []
    state_history: List[np.ndarray] = []
    share_f_history: List[float] = []

    for t in range(cfg.max_rounds + 1):
        utility_block = compute_utilities(state, params, cfg)
        share_n = float(np.mean(state == 0))
        share_i = float(np.mean(state == STATE_E))
        share_f = float(np.mean(state == STATE_F))
        c_ext = compute_externality_total(state, params, cfg)
        trajectories.append(
            {
                "t": int(t),
                "share_N": share_n,
                "share_E": share_i,
                "share_F": share_f,
                "C_ext": c_ext,
                "W": compute_social_welfare(state, params, cfg, c_star=c_star),
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

        update_agents = (
            np.asarray(update_schedule[t], dtype=int)
            if t < len(update_schedule)
            else rng.choice(cfg.n_agents, size=max(1, int(round(cfg.n_agents * cfg.async_share))), replace=False)
        )
        local_row = {
            "t": int(t),
            "subset_size": int(len(update_agents)),
            "W_subset_before": compute_social_welfare_subset(state, params, cfg, update_agents, c_star=c_star),
        }

        pre_state = state.copy()
        for idx in update_agents:
            pre_state[idx] = choose_action(
                current=int(state[idx]),
                u_n=float(utility_block["U_N"][idx]),
                u_i=float(utility_block["U_E"][idx]),
                u_f=float(utility_block["U_F"][idx]),
            )

        current_c_ext = compute_externality_total(state, params, cfg)
        pre_c_ext = compute_externality_total(pre_state, params, cfg)
        trigger = bool(pre_c_ext > c_star)
        next_state = pre_state.copy()
        charged_count = 0
        blocked_count = 0
        active_deposit = (
            float(deposit_multiple)
            * np.maximum(
                utility_block["U_F"]
                - np.maximum(utility_block["U_E"], utility_block["U_N"]),
                0.0,
            )
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
                    u_n=float(utility_block["U_N"][idx]),
                    u_i=float(utility_block["U_E"][idx]),
                    u_f=float(utility_block["U_F"][idx] - active_deposit[idx]),
                )
                if revised_action != STATE_F:
                    blocked_count += 1
                next_state[idx] = revised_action

        state = next_state
        local_row.update(
            {
                "W_subset_after": compute_social_welfare_subset(state, params, cfg, update_agents, c_star=c_star),
                "batch_threshold_trigger": float(trigger),
                "batch_current_c_ext": float(current_c_ext),
                "batch_pre_c_ext": float(pre_c_ext),
                "batch_charged_count": float(charged_count),
                "batch_blocked_count": float(blocked_count),
                "batch_mean_active_deposit": float(np.mean(active_deposit[update_agents]))
                if len(update_agents)
                else 0.0,
            }
        )
        local_rows.append(local_row)

    trajectory_df = pd.DataFrame(trajectories)
    converged = len(trajectory_df) - 1 < cfg.max_rounds
    summary = {
        "rounds_completed": int(trajectory_df.iloc[-1]["t"]),
        "converged": int(converged),
        "final_share_N": float(trajectory_df.iloc[-1]["share_N"]),
        "final_share_E": float(trajectory_df.iloc[-1]["share_E"]),
        "final_share_F": float(trajectory_df.iloc[-1]["share_F"]),
        "final_C_ext": float(trajectory_df.iloc[-1]["C_ext"]),
    }
    return {
        "trajectory": trajectory_df,
        "local_welfare": pd.DataFrame(local_rows),
        "summary": summary,
        "final_state": state.copy(),
    }


def summarize_result(
    result: Dict[str, object],
    run_id: int,
    regime: str,
    seed_share: float,
    cfg,
) -> Dict[str, object]:
    summary = result["summary"]
    trajectory = result["trajectory"]
    local = result.get("local_welfare", pd.DataFrame())
    has_safety_activity = not local.empty and "batch_charged_count" in local.columns
    steady_state_label = classify_steady_state(float(summary["final_share_F"]), bool(summary["converged"]), cfg)
    return {
        "run_id": int(run_id),
        "regime": regime,
        "seed_share_f": float(seed_share),
        "rounds_completed": int(summary["rounds_completed"]),
        "converged": int(summary["converged"]),
        "final_share_N": float(summary["final_share_N"]),
        "final_share_E": float(summary["final_share_E"]),
        "final_share_F": float(summary["final_share_F"]),
        "final_C_ext": float(summary["final_C_ext"]),
        "final_W": float(trajectory.iloc[-1]["W"]),
        "stable_window_W": float(trajectory.tail(10)["W"].mean()),
        "steady_state_label": steady_state_label,
        "charged_F_count": float(local["batch_charged_count"].sum()) if has_safety_activity else 0.0,
        "blocked_F_count": float(local["batch_blocked_count"].sum()) if has_safety_activity else 0.0,
        "threshold_trigger_rate": float(local["batch_threshold_trigger"].mean()) if has_safety_activity else 0.0,
    }


def run_experiment(
    output_dir: Path,
    max_rounds: int | None = None,
    systemic_penalty_chi: float | None = None,
    async_share: float | None = None,
) -> tuple[Dict[str, pd.DataFrame], object]:
    cfg = config_for_profile("paper", output_dir=output_dir)
    if max_rounds is not None:
        cfg.max_rounds = int(max_rounds)
    if systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = float(systemic_penalty_chi)
    if async_share is not None:
        cfg.async_share = float(async_share)
    rows: List[Dict[str, object]] = []
    trajectories: List[pd.DataFrame] = []
    meta_rows: List[Dict[str, object]] = []

    for run_id in range(cfg.n_runs):
        run_seed = cfg.seed + run_id
        rng = np.random.default_rng(run_seed)
        params = sample_agent_parameters(cfg, rng)
        seed_share = cfg.low_seed_share if run_id < cfg.n_runs // 2 else cfg.high_seed_share
        initial_state = build_initial_state(cfg, seed_share, rng=np.random.default_rng(run_seed + 1001))
        update_schedule = build_shared_update_schedule(cfg, np.random.default_rng(run_seed + 1002))

        candidate = select_safety_candidate(cfg, params)
        c_star = float(candidate["candidate_c_ext"])
        d_min_vector = build_primitive_safety_vector(candidate["state"], params, cfg, c_star)

        baseline = run_single_simulation(
            cfg,
            params,
            seed_share_f=seed_share,
            rng=np.random.default_rng(run_seed + 1),
            initial_state=initial_state,
            update_schedule=update_schedule,
            welfare_c_star=c_star,
        )
        safety_only = run_single_simulation_safety_only(
            cfg,
            params,
            seed_share_f=seed_share,
            deposit_vector=d_min_vector,
            c_star=c_star,
            rng=np.random.default_rng(run_seed + 4001),
            initial_state=initial_state,
            update_schedule=update_schedule,
        )

        rows.append(summarize_result(baseline, run_id, "primitive_baseline", seed_share, cfg))
        rows.append(summarize_result(safety_only, run_id, "safety_only", seed_share, cfg))

        for regime, result in [("primitive_baseline", baseline), ("safety_only", safety_only)]:
            traj = result["trajectory"][["t", "share_N", "share_E", "share_F", "W", "C_ext"]].copy()
            traj["run_id"] = run_id
            traj["regime"] = regime
            trajectories.append(traj)

        meta_rows.append(
            {
                "run_id": int(run_id),
                "seed_share_f": float(seed_share),
                "candidate_share_f": float(candidate["candidate_share_f"]),
                "candidate_c_ext": c_star,
                "candidate_local_eq_rate": float(candidate["candidate_local_eq_rate"]),
                "n_positive_D_min": int(np.sum(d_min_vector > 0.0)),
                "mean_positive_D_min": float(np.mean(d_min_vector[d_min_vector > 0.0]))
                if np.any(d_min_vector > 0.0)
                else 0.0,
            }
        )

    run_df = pd.DataFrame(rows)
    trajectory_df = pd.concat(trajectories, ignore_index=True)
    meta_df = pd.DataFrame(meta_rows)
    summary_df = (
        run_df.groupby("regime", as_index=False)
        .agg(
            mean_final_share_N=("final_share_N", "mean"),
            mean_final_share_E=("final_share_E", "mean"),
            mean_final_share_F=("final_share_F", "mean"),
            sd_final_share_F=("final_share_F", "std"),
            mean_final_W=("final_W", "mean"),
            sd_final_W=("final_W", "std"),
            mean_stable_window_W=("stable_window_W", "mean"),
            mean_final_C_ext=("final_C_ext", "mean"),
            mean_charged_F_count=("charged_F_count", "mean"),
            mean_blocked_F_count=("blocked_F_count", "mean"),
            mean_threshold_trigger_rate=("threshold_trigger_rate", "mean"),
            convergence_rate=("converged", "mean"),
            n_runs=("run_id", "count"),
        )
        .sort_values("regime")
    )
    data = {
        "run_summary": run_df,
        "trajectory": trajectory_df,
        "candidate_meta": meta_df,
        "summary": summary_df,
    }
    return data, cfg


def write_outputs(
    data: Dict[str, pd.DataFrame],
    output_dir: Path,
    cfg,
    main_results_dir: Path | None = None,
    timing_results_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    for name, df in data.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    summary = data["summary"].set_index("regime")
    regimes = ["primitive_baseline", "safety_only"]
    trajectory = data["trajectory"]
    max_t = int(cfg.max_rounds)
    padded_rows = []
    for (regime, run_id), sub in trajectory.groupby(["regime", "run_id"]):
        indexed = sub.sort_values("t").set_index("t")
        padded = indexed.reindex(range(max_t + 1)).ffill()
        padded["t"] = padded.index
        padded["run_id"] = int(run_id)
        padded["regime"] = regime
        padded_rows.append(padded.reset_index(drop=True))
    padded = pd.concat(padded_rows, ignore_index=True)
    padded.to_csv(output_dir / "trajectory_padded.csv", index=False)

    combined_rows = []
    for regime, label in [
        ("primitive_baseline", "PM"),
        ("safety_only", "ST"),
    ]:
        grouped = padded[padded["regime"] == regime].groupby("t", as_index=False)["W"].mean()
        grouped["mechanism_label"] = label
        combined_rows.append(grouped)

    if main_results_dir is not None:
        main_welfare_path = main_results_dir / "welfare_trajectory_comparison.csv"
        if main_welfare_path.exists():
            main_welfare = pd.read_csv(main_welfare_path)

            def append_main_path(source: str, mechanism: str, label: str) -> None:
                selected = main_welfare[
                    (main_welfare["source"] == source)
                    & (main_welfare["mechanism"] == mechanism)
                ].copy()
                if selected.empty:
                    return
                max_t_combined = max(max_t, int(selected["t"].max()))
                padded_selected_rows = []
                for run_id, sub in selected.groupby("run_id"):
                    indexed = sub.sort_values("t").set_index("t")[["W"]]
                    padded_selected = indexed.reindex(range(max_t_combined + 1)).ffill()
                    padded_selected["t"] = padded_selected.index
                    padded_selected["run_id"] = int(run_id)
                    padded_selected_rows.append(padded_selected.reset_index(drop=True))
                padded_selected_df = pd.concat(padded_selected_rows, ignore_index=True)
                grouped = padded_selected_df.groupby("t", as_index=False)["W"].mean()
                grouped["mechanism_label"] = label
                combined_rows.append(grouped)

            append_main_path("simulation_rhood", "rhood", "RR")
            append_main_path("simulation_deposit", "deposit", "DM")

    if timing_results_dir is not None:
        timing_path = timing_results_dir / "trajectory.csv"
        if timing_path.exists():
            timing_df = pd.read_csv(timing_path)
            selected = timing_df[timing_df["intervention_label"] == "t=20"].copy()
            if not selected.empty:
                max_t_combined = max(max_t, int(selected["t"].max()))
                padded_selected_rows = []
                for run_id, sub in selected.groupby("run_id"):
                    indexed = sub.sort_values("t").set_index("t")[["W"]]
                    padded_selected = indexed.reindex(range(max_t_combined + 1)).ffill()
                    padded_selected["t"] = padded_selected.index
                    padded_selected["run_id"] = int(run_id)
                    padded_selected_rows.append(padded_selected.reset_index(drop=True))
                padded_selected_df = pd.concat(padded_selected_rows, ignore_index=True)
                grouped = padded_selected_df.groupby("t", as_index=False)["W"].mean()
                grouped["mechanism_label"] = "DM-ASD (t=20)"
                combined_rows.append(grouped)

    combined = pd.concat(combined_rows, ignore_index=True)
    combined_padded_rows = []
    for label, sub in combined.groupby("mechanism_label"):
        indexed = sub.sort_values("t").set_index("t")[["W"]]
        padded_combined = indexed.reindex(range(cfg.max_rounds + 1)).ffill()
        padded_combined["t"] = padded_combined.index
        padded_combined["mechanism_label"] = label
        combined_padded_rows.append(padded_combined.reset_index(drop=True))
    combined = pd.concat(combined_padded_rows, ignore_index=True)
    combined.to_csv(output_dir / "welfare_trajectory_with_redistribution_safety.csv", index=False)
    combined.to_csv(output_dir / "welfare_trajectory_mechanism_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    combined_styles = {
        "PM": ("#8172B2", "PM"),
        "RR": ("#C44E52", "RR"),
        "ST": ("#4C72B0", "ST"),
        "DM": ("#55A868", "DM"),
        "DM-ASD (t=20)": ("#E5AE38", "DM-ASD (t=20)"),
    }
    for label, (color, display_label) in combined_styles.items():
        sub = combined[combined["mechanism_label"] == label]
        if sub.empty:
            continue
        ax.plot(sub["t"], sub["W"], linewidth=2.0, color=color, label=display_label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean social welfare")
    ax.set_title("Welfare trajectories by mechanism")
    ax.set_xlim(0, cfg.max_rounds)
    ax.set_ylim(900, 1900)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "welfare_trajectory_with_redistribution_safety.png", dpi=220)
    fig.savefig(figures / "welfare_trajectory_mechanism_ablation.png", dpi=220)
    plt.close(fig)

    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Safety-Only Experiment\n\n")
        f.write("This experiment uses the retained paper profile and the same initial-state/update-schedule construction as the main mechanism experiments. ")
        f.write("It compares the primitive three-action baseline against a safety-threshold rule applied without the redistribution mechanism.\n\n")
        f.write("The safety boundary is selected from the same candidate-state grid used in the main experiments. ")
        f.write("The active `D_i^min` is recomputed from current primitive utilities before every triggered update; ")
        f.write("the candidate state is used only to define `C*`.\n\n")
        f.write("Files:\n")
        f.write("- `summary.csv`: averaged final shares, welfare, externality, and mechanism activity by regime.\n")
        f.write("- `run_summary.csv`: per-run outcomes.\n")
        f.write("- `trajectory.csv`: raw per-run trajectories.\n")
        f.write("- `trajectory_padded.csv`: padded trajectories for full-horizon comparison.\n")
        f.write("- `candidate_meta.csv`: selected safety boundary and primitive D-min metadata.\n")
        f.write("- `welfare_trajectory_mechanism_ablation.csv` and `figures/welfare_trajectory_mechanism_ablation.png`: combined welfare paths for the no-mechanism primitive baseline, safety-only ablation, redistribution baseline, and redistribution plus safety-threshold extension.\n")
        f.write("- `welfare_trajectory_with_redistribution_safety.csv` and `figures/welfare_trajectory_with_redistribution_safety.png`: same combined mechanism-ablation output, retained for backward compatibility.\n")
        if main_results_dir is not None:
            f.write(f"\nRedistribution plus safety-threshold data source: `{main_results_dir}`.\n")
        if timing_results_dir is not None:
            f.write(f"DM-ASD timing source: `{timing_results_dir}`.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "safety-only-experiment",
    )
    parser.add_argument(
        "--main-results-dir",
        type=Path,
        default=ROOT / "outputs" / "complete-experiment",
    )
    parser.add_argument(
        "--timing-results-dir",
        type=Path,
        default=ROOT / "outputs" / "paper-main-final" / "timing",
    )
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--systemic-penalty-chi", type=float, default=None)
    parser.add_argument("--async-share", type=float, default=None)
    args = parser.parse_args()
    data, cfg = run_experiment(
        args.output_dir,
        max_rounds=args.max_rounds,
        systemic_penalty_chi=args.systemic_penalty_chi,
        async_share=args.async_share,
    )
    write_outputs(
        data,
        args.output_dir,
        cfg,
        main_results_dir=args.main_results_dir,
        timing_results_dir=args.timing_results_dir,
    )
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2)
    print(f"wrote outputs to {args.output_dir}")
    print(data["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
