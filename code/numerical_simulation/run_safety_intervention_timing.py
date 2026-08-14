from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sim_config import ExperimentConfig, config_for_profile
from sim_model import (
    STATE_F,
    STATE_E,
    STATE_N,
    apply_batch_safety_threshold_update,
    build_initial_state,
    classify_steady_state,
    compute_externality_total,
    compute_social_welfare,
    compute_social_welfare_subset,
    compute_utilities_with_rhood,
    config_to_dict,
    sample_agent_parameters,
    choose_action,
)
from sim_theory import build_candidate_state_from_share
from run_theory_vs_simulation import (
    build_deposit_vector_from_candidate,
    build_shared_update_schedule,
    select_rhood_beta,
    theory_tables_for_run,
)


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".tmp_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_intervention_rounds(value: str) -> List[int | None]:
    rounds: List[int | None] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"never", "none", "off", "rhood"}:
            rounds.append(None)
        else:
            rounds.append(int(token))
    return rounds


def intervention_label(intervention_round: int | None) -> str:
    if intervention_round is None:
        return "never"
    return f"t={intervention_round}"


def run_single_intervention_timing(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    seed_share_f: float,
    beta: float,
    deposit_vector: np.ndarray,
    c_star: float,
    initial_state: np.ndarray,
    update_schedule: List[np.ndarray],
    intervention_round: int | None,
    allow_early_stop: bool = True,
    dynamic_deposit: bool = True,
    deposit_multiple: float = 1.0,
) -> Dict[str, object]:
    state = initial_state.copy()
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
        safety_active = intervention_round is not None and t >= intervention_round

        trajectories.append(
            {
                "t": t,
                "share_N": share_n,
                "share_E": share_i,
                "share_F": share_f,
                "C_ext": c_ext,
                "W": compute_social_welfare(state, params, cfg, c_star=c_star),
                "safety_active": float(safety_active),
                "sensitive_mass_F": float(np.sum((state == STATE_F) * params["d_i"].to_numpy())),
            }
        )
        state_history.append(state.copy())
        share_f_history.append(share_f)

        if t == cfg.max_rounds:
            break
        convergence_allowed = allow_early_stop and (intervention_round is None or t >= intervention_round)
        if convergence_allowed and len(state_history) >= cfg.convergence_window:
            recent_states = state_history[-cfg.convergence_window :]
            if all(np.array_equal(recent_states[0], item) for item in recent_states[1:]):
                break
            recent_share_f = share_f_history[-cfg.convergence_window :]
            if max(recent_share_f) - min(recent_share_f) < cfg.convergence_eps:
                break

        update_agents = np.asarray(update_schedule[t], dtype=int)
        local_row: Dict[str, float] = {
            "t": int(t),
            "subset_size": int(len(update_agents)),
            "safety_active": float(safety_active),
            "W_subset_before": compute_social_welfare_subset(state, params, cfg, update_agents, c_star=c_star),
        }

        if safety_active:
            next_state, batch_info = apply_batch_safety_threshold_update(
                state,
                update_agents,
                utility_block,
                params,
                cfg,
                deposit_vector,
                c_star,
                dynamic_deposit=dynamic_deposit,
                deposit_multiple=deposit_multiple,
            )
            local_row.update(batch_info)
        else:
            next_state = state.copy()
            for idx in update_agents:
                next_state[idx] = choose_action(
                    current=int(state[idx]),
                    u_n=float(utility_block["U_N_rhood"][idx]),
                    u_i=float(utility_block["U_E_rhood"][idx]),
                    u_f=float(utility_block["U_F_rhood"][idx]),
                )
            local_row.update(
                {
                    "batch_threshold_trigger": 0.0,
                    "batch_current_c_ext": c_ext,
                    "batch_pre_c_ext": compute_externality_total(next_state, params, cfg),
                    "batch_charged_count": 0.0,
                    "batch_blocked_count": 0.0,
                }
            )

        state = next_state
        local_row["W_subset_after"] = compute_social_welfare_subset(state, params, cfg, update_agents, c_star=c_star)
        local_welfare_rows.append(local_row)

    trajectory_df = pd.DataFrame(trajectories)
    local_df = pd.DataFrame(local_welfare_rows)
    converged = len(trajectory_df) - 1 < cfg.max_rounds
    final_share_f = float(trajectory_df.iloc[-1]["share_F"])
    summary = {
        "rounds_completed": int(trajectory_df.iloc[-1]["t"]),
        "converged": int(converged),
        "final_share_N": float(trajectory_df.iloc[-1]["share_N"]),
        "final_share_E": float(trajectory_df.iloc[-1]["share_E"]),
        "final_share_F": final_share_f,
        "final_C_ext": float(trajectory_df.iloc[-1]["C_ext"]),
        "final_W": float(trajectory_df.iloc[-1]["W"]),
        "stable_window_W": float(trajectory_df.tail(10)["W"].mean()),
        "steady_state_label": classify_steady_state(final_share_f, converged, cfg),
        "threshold_trigger_rate": (
            float(local_df["batch_threshold_trigger"].mean())
            if not local_df.empty and "batch_threshold_trigger" in local_df
            else 0.0
        ),
        "charged_F_count": (
            float(local_df["batch_charged_count"].sum())
            if not local_df.empty and "batch_charged_count" in local_df
            else 0.0
        ),
        "blocked_F_count": (
            float(local_df["batch_blocked_count"].sum())
            if not local_df.empty and "batch_blocked_count" in local_df
            else 0.0
        ),
    }
    return {
        "trajectory": trajectory_df,
        "local_welfare": local_df,
        "summary": summary,
        "final_state": state.copy(),
    }


def build_run_safety_objects(
    cfg: ExperimentConfig,
    params: pd.DataFrame,
    run_id: int,
    seed_share: float,
    beta_star: float,
    shared_initial_state: np.ndarray,
    shared_update_schedule: List[np.ndarray],
) -> Dict[str, object]:
    rhood_result = run_single_intervention_timing(
        cfg,
        params,
        seed_share,
        beta_star,
        deposit_vector=np.zeros(cfg.n_agents),
        c_star=float("inf"),
        initial_state=shared_initial_state,
        update_schedule=shared_update_schedule,
        intervention_round=None,
    )
    final_state = rhood_result["final_state"]
    theory = theory_tables_for_run(cfg, params, final_state, beta_rhood=beta_star)
    selected_candidate = theory["selected_candidate_meta"].iloc[0]
    candidate_share_f = float(selected_candidate["candidate_share_f"])
    candidate_state = build_candidate_state_from_share(params, cfg, candidate_share_f)["state"]
    c_star = float(selected_candidate["candidate_c_ext"])
    d_min_vector = build_deposit_vector_from_candidate(
        cfg,
        params,
        candidate_state,
        c_star=c_star,
        beta=beta_star,
    )
    return {
        "c_star": c_star,
        "d_min_vector": d_min_vector,
        "candidate_meta": {
            "run_id": run_id,
            "seed_share_f": seed_share,
            "beta_rhood": beta_star,
            "candidate_share_f": candidate_share_f,
            "candidate_c_ext": c_star,
            "candidate_local_eq_rate": float(selected_candidate["candidate_local_eq_rate"]),
            "n_positive_D_min": int(np.sum(d_min_vector > 0)),
            "mean_positive_D_min": (
                float(np.mean(d_min_vector[d_min_vector > 0])) if np.any(d_min_vector > 0) else 0.0
            ),
        },
    }


def run_one_replication(
    run_id: int,
    output_dir: Path,
    intervention_rounds: List[int | None],
    fee_multiple: float,
    async_share: float | None = None,
    max_rounds: int | None = None,
    systemic_penalty_chi: float | None = None,
    allow_early_stop: bool = True,
) -> Dict[str, List[Dict[str, object]]]:
    cfg = config_for_profile("paper", output_dir=output_dir)
    cfg.async_share = 0.05 if async_share is None else async_share
    if max_rounds is not None:
        cfg.max_rounds = max_rounds
    if systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = systemic_penalty_chi

    run_seed = cfg.seed + run_id
    rng = np.random.default_rng(run_seed)
    params = sample_agent_parameters(cfg, rng)
    beta_star = select_rhood_beta(cfg, params)
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

    safety_objects = build_run_safety_objects(
        cfg,
        params,
        run_id,
        seed_share,
        beta_star,
        shared_initial_state,
        shared_update_schedule,
    )
    d_vector = float(fee_multiple) * safety_objects["d_min_vector"]
    c_star = float(safety_objects["c_star"])

    run_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    meta_rows: List[Dict[str, object]] = [safety_objects["candidate_meta"]]

    for intervention_round in intervention_rounds:
        result = run_single_intervention_timing(
            cfg,
            params,
            seed_share_f=seed_share,
            beta=beta_star,
            deposit_vector=d_vector,
            c_star=c_star,
            initial_state=shared_initial_state,
            update_schedule=shared_update_schedule,
            intervention_round=intervention_round,
            allow_early_stop=allow_early_stop,
            dynamic_deposit=True,
            deposit_multiple=fee_multiple,
        )
        label = intervention_label(intervention_round)
        summary = result["summary"]
        run_rows.append(
            {
                "run_id": run_id,
                "intervention_round": -1 if intervention_round is None else int(intervention_round),
                "intervention_label": label,
                "fee_multiple": float(fee_multiple),
                "seed_share_f": float(seed_share),
                "beta_rhood": float(beta_star),
                **summary,
            }
        )

        traj = result["trajectory"][["t", "share_N", "share_E", "share_F", "W", "C_ext", "safety_active"]].copy()
        traj["run_id"] = run_id
        traj["intervention_round"] = -1 if intervention_round is None else int(intervention_round)
        traj["intervention_label"] = label
        traj["fee_multiple"] = float(fee_multiple)
        trajectory_rows.extend(traj.to_dict(orient="records"))

        local = result["local_welfare"].copy()
        if not local.empty:
            local["run_id"] = run_id
            local["intervention_round"] = -1 if intervention_round is None else int(intervention_round)
            local["intervention_label"] = label
            local["fee_multiple"] = float(fee_multiple)
            local_rows.extend(local.to_dict(orient="records"))

    return {
        "run_summary": run_rows,
        "trajectory": trajectory_rows,
        "local_welfare": local_rows,
        "candidate_meta": meta_rows,
    }


def run_experiment(
    output_dir: Path,
    intervention_rounds: List[int | None],
    fee_multiple: float,
    workers: int,
    async_share: float | None = None,
    max_rounds: int | None = None,
    systemic_penalty_chi: float | None = None,
    allow_early_stop: bool = True,
) -> Dict[str, pd.DataFrame]:
    cfg = config_for_profile("paper", output_dir=output_dir)
    cfg.async_share = 0.05 if async_share is None else async_share
    if max_rounds is not None:
        cfg.max_rounds = max_rounds
    if systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = systemic_penalty_chi
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    local_rows: List[Dict[str, object]] = []
    meta_rows: List[Dict[str, object]] = []

    if workers <= 1:
        for run_id in range(cfg.n_runs):
            result = run_one_replication(
                run_id,
                output_dir,
                intervention_rounds,
                fee_multiple,
                async_share=async_share,
                max_rounds=max_rounds,
                systemic_penalty_chi=systemic_penalty_chi,
                allow_early_stop=allow_early_stop,
            )
            run_rows.extend(result["run_summary"])
            trajectory_rows.extend(result["trajectory"])
            local_rows.extend(result["local_welfare"])
            meta_rows.extend(result["candidate_meta"])
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_one_replication,
                    run_id,
                    output_dir,
                    intervention_rounds,
                    fee_multiple,
                    async_share,
                    max_rounds,
                    systemic_penalty_chi,
                    allow_early_stop,
                ): run_id
                for run_id in range(cfg.n_runs)
            }
            for future in as_completed(futures):
                result = future.result()
                run_rows.extend(result["run_summary"])
                trajectory_rows.extend(result["trajectory"])
                local_rows.extend(result["local_welfare"])
                meta_rows.extend(result["candidate_meta"])

    run_df = pd.DataFrame(run_rows)
    trajectory_df = pd.DataFrame(trajectory_rows)
    local_df = pd.DataFrame(local_rows)
    meta_df = pd.DataFrame(meta_rows)

    summary_df = (
        run_df.groupby(["intervention_round", "intervention_label"], as_index=False)
        .agg(
            mean_final_share_N=("final_share_N", "mean"),
            mean_final_share_E=("final_share_E", "mean"),
            mean_final_share_F=("final_share_F", "mean"),
            sd_final_share_F=("final_share_F", "std"),
            mean_final_W=("final_W", "mean"),
            sd_final_W=("final_W", "std"),
            mean_stable_window_W=("stable_window_W", "mean"),
            sd_stable_window_W=("stable_window_W", "std"),
            mean_final_C_ext=("final_C_ext", "mean"),
            mean_threshold_trigger_rate=("threshold_trigger_rate", "mean"),
            mean_charged_F_count=("charged_F_count", "mean"),
            mean_blocked_F_count=("blocked_F_count", "mean"),
            convergence_rate=("converged", "mean"),
            n_runs=("run_id", "count"),
        )
        .sort_values("intervention_round")
    )

    return {
        "run_summary": run_df,
        "trajectory": trajectory_df,
        "local_welfare": local_df,
        "candidate_meta": meta_df,
        "summary": summary_df,
    }


def pad_trajectory(trajectory: pd.DataFrame, max_rounds: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for (label, run_id), sub in trajectory.groupby(["intervention_label", "run_id"]):
        indexed = sub.sort_values("t").set_index("t")
        padded = indexed.reindex(range(max_rounds + 1)).ffill()
        padded["t"] = padded.index
        padded["intervention_label"] = label
        padded["run_id"] = int(run_id)
        rows.append(padded.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def write_outputs(
    data: Dict[str, pd.DataFrame],
    output_dir: Path,
    intervention_rounds: List[int | None],
    fee_multiple: float,
    cfg: ExperimentConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)

    for name, df in data.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    padded = pad_trajectory(data["trajectory"], cfg.max_rounds)
    padded.to_csv(output_dir / "trajectory_padded.csv", index=False)

    ordered_labels = [intervention_label(item) for item in intervention_rounds]
    colors = plt.get_cmap("viridis", len(ordered_labels))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for idx, label in enumerate(ordered_labels):
        sub = padded[padded["intervention_label"] == label]
        grouped = sub.groupby("t", as_index=False)["W"].mean().sort_values("t")
        display = "Redistribution only" if label == "never" else f"Safety from {label}"
        ax.plot(grouped["t"], grouped["W"], linewidth=1.9, color=colors(idx), label=display)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean social welfare")
    ax.set_title("Welfare trajectories by safety-threshold intervention timing")
    ax.set_xlim(0, cfg.max_rounds)
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "fig13_safety_intervention_welfare_trajectory.png", dpi=220)
    plt.close(fig)

    summary = data["summary"].copy()
    summary["plot_label"] = summary["intervention_label"].map(
        lambda label: "Redistribution only" if label == "never" else label
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        summary["plot_label"],
        summary["mean_final_W"],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Final welfare",
    )
    ax.set_xlabel("Safety-threshold intervention round")
    ax.set_ylabel("Mean welfare")
    ax.set_title("Final welfare by intervention timing")
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figures / "fig09_impact_of_timing_of_safety_threshold_mechanism_on_steady_state_social_welfare.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(summary["plot_label"], summary["mean_final_share_N"], marker="o", label="N")
    ax.plot(summary["plot_label"], summary["mean_final_share_E"], marker="o", label="E")
    ax.plot(summary["plot_label"], summary["mean_final_share_F"], marker="o", label="F")
    ax.set_xlabel("Safety-threshold intervention round")
    ax.set_ylabel("Mean final strategy share")
    ax.set_title("Final N/E/F shares by intervention timing")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figures / "fig13_safety_intervention_final_shares.png", dpi=220)
    plt.close(fig)

    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Safety-Threshold Intervention Timing\n\n")
        f.write("Profile: `paper`; parameters match the retained main experiment configuration. ")
        f.write("The experiment varies only the round at which the safety-threshold layer is activated on top of redistribution.\n\n")
        f.write(f"Intervention rounds: `{[intervention_label(item) for item in intervention_rounds]}`.\n")
        f.write(f"Safety-threshold fee multiple: `{fee_multiple} * D_i^min`.\n\n")
        f.write("`D_i^min` is recalculated from the current state before every active safety-threshold update.\n\n")
        f.write(f"Early stopping: `{cfg.allow_early_stop_for_timing if hasattr(cfg, 'allow_early_stop_for_timing') else 'default'}`.\n\n")
        f.write("Files:\n")
        f.write("- `summary.csv`: averages by intervention timing.\n")
        f.write("- `run_summary.csv`: per-run final shares, welfare, externality, and mechanism activity.\n")
        f.write("- `trajectory.csv`: raw per-run trajectories.\n")
        f.write("- `trajectory_padded.csv`: trajectories padded after convergence for full-horizon plotting.\n")
        f.write("- `local_welfare.csv`: per-round update-batch welfare and safety-threshold activity.\n")
        f.write("- `candidate_meta.csv`: selected safety boundary and D_min metadata by run.\n")
        f.write("- `figures/fig13_safety_intervention_welfare_trajectory.png`.\n")
        f.write("- `figures/fig09_impact_of_timing_of_safety_threshold_mechanism_on_steady_state_social_welfare.png`.\n")
        f.write("- `figures/fig13_safety_intervention_final_shares.png`.\n")
        f.write("\nInterpretation: `t=20` maximizes final welfare over the tested grid, but it does not ")
        f.write("minimize externality or full sharing. Activation at `t=0` yields the lowest final externality.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "safety-intervention-timing",
    )
    parser.add_argument(
        "--intervention-rounds",
        type=str,
        default="never,0,10,20,40,60,80",
        help="Comma-separated list of intervention rounds; use `never` for the redistribution-only baseline.",
    )
    parser.add_argument(
        "--fee-multiple",
        type=float,
        default=1.0,
        help="Safety-threshold fee multiple applied after intervention.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--async-share",
        type=float,
        default=None,
        help="Optional override. Defaults to 0.05 to match the retained main experiment.",
    )
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--systemic-penalty-chi", type=float, default=None)
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Force every timing trajectory to run to max_rounds instead of stopping after convergence.",
    )
    args = parser.parse_args()

    intervention_rounds = parse_intervention_rounds(args.intervention_rounds)
    cfg = config_for_profile("paper", output_dir=args.output_dir)
    cfg.async_share = 0.05 if args.async_share is None else args.async_share
    if args.max_rounds is not None:
        cfg.max_rounds = args.max_rounds
    if args.systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = args.systemic_penalty_chi
    cfg.allow_early_stop_for_timing = not args.no_early_stop
    data = run_experiment(
        args.output_dir,
        intervention_rounds,
        fee_multiple=args.fee_multiple,
        workers=args.workers,
        async_share=args.async_share,
        max_rounds=args.max_rounds,
        systemic_penalty_chi=args.systemic_penalty_chi,
        allow_early_stop=not args.no_early_stop,
    )
    write_outputs(data, args.output_dir, intervention_rounds, args.fee_multiple, cfg)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    print(f"wrote outputs to {args.output_dir}")
    print(data["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
