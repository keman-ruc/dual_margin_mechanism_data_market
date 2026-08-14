from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sim_config import config_for_profile
from sim_model import STATE_F, STATE_E, STATE_N, build_initial_state, config_to_dict, sample_agent_parameters
from sim_runner import run_single_simulation, run_single_simulation_with_deposit
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


def final_row(result: Dict[str, object], run_id: int, multiple: float, beta: float, seed_share: float) -> Dict[str, object]:
    summary = result["summary"]
    trajectory = result["trajectory"]
    final_w = float(trajectory.iloc[-1]["W"])
    stable_w = float(trajectory.tail(10)["W"].mean())
    local = result.get("local_welfare", pd.DataFrame())
    charged = float(local["batch_charged_count"].sum()) if not local.empty and "batch_charged_count" in local else 0.0
    blocked = float(local["batch_blocked_count"].sum()) if not local.empty and "batch_blocked_count" in local else 0.0
    trigger_rate = float(local["batch_threshold_trigger"].mean()) if not local.empty and "batch_threshold_trigger" in local else 0.0
    return {
        "run_id": run_id,
        "fee_multiple": multiple,
        "seed_share_f": seed_share,
        "beta_rhood": beta,
        "rounds_completed": int(summary["rounds_completed"]),
        "converged": int(summary["converged"]),
        "final_share_N": float(summary["final_share_N"]),
        "final_share_E": float(summary["final_share_E"]),
        "final_share_F": float(summary["final_share_F"]),
        "final_C_ext": float(summary["final_C_ext"]),
        "final_W": final_w,
        "stable_window_W": stable_w,
        "charged_F_count": charged,
        "blocked_F_count": blocked,
        "threshold_trigger_rate": trigger_rate,
    }


def run_one_replication(
    run_id: int,
    output_dir: Path,
    fee_grid: List[float],
    async_share: float | None = None,
    max_rounds: int | None = None,
    systemic_penalty_chi: float | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    cfg = config_for_profile("paper", output_dir=output_dir)
    if async_share is not None:
        cfg.async_share = async_share
    if max_rounds is not None:
        cfg.max_rounds = max_rounds
    if systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = systemic_penalty_chi
    run_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    meta_rows: List[Dict[str, object]] = []

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

    baseline_result = run_single_simulation(
        cfg,
        params,
        seed_share_f=seed_share,
        rng=np.random.default_rng(run_seed + 1),
        initial_state=shared_initial_state,
        update_schedule=shared_update_schedule,
    )
    final_state = baseline_result["final_state"]
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

    meta_rows.append(
        {
            "run_id": run_id,
            "seed_share_f": seed_share,
            "beta_rhood": beta_star,
            "candidate_share_f": candidate_share_f,
            "candidate_c_ext": c_star,
            "candidate_local_eq_rate": float(selected_candidate["candidate_local_eq_rate"]),
            "n_positive_D_min": int(np.sum(d_min_vector > 0)),
            "mean_positive_D_min": float(np.mean(d_min_vector[d_min_vector > 0])) if np.any(d_min_vector > 0) else 0.0,
        }
    )

    for multiple in fee_grid:
        deposit_vector = float(multiple) * d_min_vector
        result = run_single_simulation_with_deposit(
            cfg,
            params,
            seed_share_f=seed_share,
            beta=beta_star,
            deposit_vector=deposit_vector,
            c_star=c_star,
            rng=np.random.default_rng(run_seed + 4000 + int(round(multiple * 100))),
            initial_state=shared_initial_state,
            update_schedule=shared_update_schedule,
            deposit_multiple=float(multiple),
            dynamic_deposit=True,
        )
        run_rows.append(final_row(result, run_id, float(multiple), beta_star, seed_share))
        traj = result["trajectory"][["t", "share_N", "share_E", "share_F", "W", "C_ext"]].copy()
        traj["run_id"] = run_id
        traj["fee_multiple"] = float(multiple)
        trajectory_rows.extend(traj.to_dict(orient="records"))

    return {
        "run_summary": run_rows,
        "trajectory": trajectory_rows,
        "candidate_meta": meta_rows,
    }


def run_experiment(
    output_dir: Path,
    fee_grid: List[float],
    workers: int,
    async_share: float | None = None,
    max_rounds: int | None = None,
    systemic_penalty_chi: float | None = None,
) -> Dict[str, pd.DataFrame]:
    cfg = config_for_profile("paper", output_dir=output_dir)
    if async_share is not None:
        cfg.async_share = async_share
    if max_rounds is not None:
        cfg.max_rounds = max_rounds
    if systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = systemic_penalty_chi
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []
    meta_rows: List[Dict[str, object]] = []

    if workers <= 1:
        for run_id in range(cfg.n_runs):
            result = run_one_replication(
                run_id,
                output_dir,
                fee_grid,
                async_share=async_share,
                max_rounds=max_rounds,
                systemic_penalty_chi=systemic_penalty_chi,
            )
            run_rows.extend(result["run_summary"])
            trajectory_rows.extend(result["trajectory"])
            meta_rows.extend(result["candidate_meta"])
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_one_replication,
                    run_id,
                    output_dir,
                    fee_grid,
                    async_share,
                    max_rounds,
                    systemic_penalty_chi,
                ): run_id
                for run_id in range(cfg.n_runs)
            }
            for future in as_completed(futures):
                result = future.result()
                run_rows.extend(result["run_summary"])
                trajectory_rows.extend(result["trajectory"])
                meta_rows.extend(result["candidate_meta"])

    run_df = pd.DataFrame(run_rows)
    trajectory_df = pd.DataFrame(trajectory_rows)
    meta_df = pd.DataFrame(meta_rows)

    summary_df = (
        run_df.groupby("fee_multiple", as_index=False)
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
            mean_charged_F_count=("charged_F_count", "mean"),
            mean_blocked_F_count=("blocked_F_count", "mean"),
            mean_threshold_trigger_rate=("threshold_trigger_rate", "mean"),
            convergence_rate=("converged", "mean"),
            n_runs=("run_id", "count"),
        )
        .sort_values("fee_multiple")
    )

    return {
        "run_summary": run_df,
        "trajectory": trajectory_df,
        "candidate_meta": meta_df,
        "summary": summary_df,
    }


def write_outputs(data: Dict[str, pd.DataFrame], output_dir: Path, fee_grid: List[float], max_rounds: int) -> None:
    for name, df in data.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(summary["fee_multiple"], summary["mean_final_W"], marker="o", label="Final welfare")
    ax.plot(summary["fee_multiple"], summary["mean_stable_window_W"], marker="s", label="Stable-window welfare")
    ax.axvline(1.0, linestyle="--", color="#C44E52", linewidth=1.2, label="D / D_min = 1")
    ax.axvspan(1.5, max(fee_grid), color="#999999", alpha=0.10, label="No convergence")
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    ax.set_xticks(fee_grid)
    ax.set_xticklabels([f"{value:g}" for value in fee_grid])
    ax.set_xlabel("Safety-threshold fee multiple D / D_min")
    ax.set_ylabel("Mean welfare")
    ax.set_title("Welfare under safety-threshold fee levels")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig12_safety_fee_welfare_summary.png", dpi=220)
    plt.close(fig)

    trajectory = data["trajectory"]
    padded_rows: List[pd.DataFrame] = []
    for (multiple, run_id), sub in trajectory.groupby(["fee_multiple", "run_id"]):
        indexed = sub.sort_values("t").set_index("t")
        padded = indexed.reindex(range(max_rounds + 1)).ffill()
        padded["t"] = padded.index
        padded["fee_multiple"] = float(multiple)
        padded["run_id"] = int(run_id)
        padded_rows.append(padded.reset_index(drop=True))
    padded_trajectory = pd.concat(padded_rows, ignore_index=True)
    padded_trajectory.to_csv(output_dir / "trajectory_padded.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    fee_values = sorted(padded_trajectory["fee_multiple"].unique())
    cmap = plt.get_cmap("viridis", len(fee_values))
    for idx, multiple in enumerate(fee_values):
        sub = padded_trajectory[padded_trajectory["fee_multiple"] == multiple]
        grouped = sub.groupby("t", as_index=False)["W"].mean().sort_values("t")
        label = "D/Dmin=0" if np.isclose(multiple, 0.0) else f"D/Dmin={multiple:g}"
        ax.plot(grouped["t"], grouped["W"], linewidth=1.8, color=cmap(idx), label=label)
    ax.axvline(1.0, linestyle="--", color="#C44E52", linewidth=1.2, alpha=0.0)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean social welfare")
    ax.set_title("Welfare trajectories under safety-threshold fee levels")
    ax.set_xlim(0, max_rounds)
    ax.set_ylim(900, 1900)
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "fig12_social_welfare_trajectories_under_different_safety_threshold_fee_multiples.png", dpi=220)
    plt.close(fig)

    with (output_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("# Safety-Threshold Fee Sensitivity\n\n")
        f.write("Profile: `paper`; all model parameters follow the retained paper configuration. ")
        f.write("The only varied parameter is the safety-threshold fee multiple `D_i / D_i^{min}`.\n\n")
        f.write("`D_i^{min}` is recalculated from the current state before every update. ")
        f.write("The multiple therefore scales a dynamic fee rather than a fixed candidate-state vector.\n\n")
        f.write(f"Fee grid: `{fee_grid}`.\n\n")
        f.write("Files:\n")
        f.write("- `summary.csv`: averages by fee multiple.\n")
        f.write("- `run_summary.csv`: per-run final N/E/F shares, welfare, externality, and charged F-choice counts.\n")
        f.write("- `trajectory.csv`: per-run trajectories for shares, welfare, and externality.\n")
        f.write("- `trajectory_padded.csv`: per-run trajectories padded after convergence for full-horizon plotting.\n")
        f.write("- `candidate_meta.csv`: selected safety boundary and D_min metadata by run.\n")
        f.write("- `figures/fig12_safety_fee_welfare_summary.png`: final and stable-window welfare by fee multiple.\n")
        f.write("- `figures/fig12_social_welfare_trajectories_under_different_safety_threshold_fee_multiples.png`: welfare trajectories by fee multiple.\n")
        f.write("\nInterpretation: `D/D_min = 1` is the effective threshold in this grid. ")
        f.write("Runs above `1` do not converge within the simulated horizon and show persistent oscillation; ")
        f.write("their final-round values are not steady-state estimates.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "safety-threshold-fee-sensitivity",
    )
    parser.add_argument(
        "--fee-grid",
        type=str,
        default="0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--async-share",
        type=float,
        default=None,
        help="Optional override for the paper profile asynchronous update share.",
    )
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--systemic-penalty-chi", type=float, default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures and README from CSV files already present in --output-dir.",
    )
    args = parser.parse_args()

    fee_grid = [float(item.strip()) for item in args.fee_grid.split(",") if item.strip()]
    if args.plot_only:
        data = {
            name: pd.read_csv(args.output_dir / f"{name}.csv")
            for name in ["run_summary", "trajectory", "candidate_meta", "summary"]
        }
        config_path = args.output_dir / "config.json"
        max_rounds = 100
        if config_path.exists():
            max_rounds = int(json.loads(config_path.read_text(encoding="utf-8"))["max_rounds"])
        fee_grid = sorted(float(value) for value in data["summary"]["fee_multiple"].unique())
        write_outputs(data, args.output_dir, fee_grid, max_rounds=max_rounds)
        print(f"regenerated figures from existing outputs in {args.output_dir}")
        return

    data = run_experiment(
        args.output_dir,
        fee_grid,
        workers=args.workers,
        async_share=args.async_share,
        max_rounds=args.max_rounds,
        systemic_penalty_chi=args.systemic_penalty_chi,
    )

    cfg = config_for_profile("paper", output_dir=args.output_dir)
    if args.async_share is not None:
        cfg.async_share = args.async_share
    if args.max_rounds is not None:
        cfg.max_rounds = args.max_rounds
    if args.systemic_penalty_chi is not None:
        cfg.systemic_penalty_chi = args.systemic_penalty_chi
    cfg.deposit_multipliers = tuple(fee_grid)
    write_outputs(data, args.output_dir, fee_grid, max_rounds=cfg.max_rounds)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2)

    print(f"wrote outputs to {args.output_dir}")
    print(data["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
