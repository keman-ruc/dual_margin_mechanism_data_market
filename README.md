# Balancing Participation and Privacy: An Adaptive Dual-Margin Mechanism to Mitigate Data Externalities in Cooperative Data Markets

Code and data for the ICDE 2027 submission.

## Structure

- `code/numerical_simulation/` — simulation implementation (`sim_model.py`, `sim_theory.py`, `sim_config.py`) and experiment scripts (`run_theory_vs_simulation.py`, `run_safety_fee_sensitivity.py`, `run_safety_intervention_timing.py`, `run_safety_only_experiment.py`).
- `results/data/main/` — main experiment outputs: proposition validation, redistribution feasibility, and mechanism comparison.
- `results/data/fee/` — safety-threshold fee-multiple sensitivity.
- `results/data/timing/` — safety-threshold intervention-timing sensitivity.
- `results/data/ablation/` — safety-only ablation (no redistribution).

## Reproduce

```bash
cd code/numerical_simulation
python3 run_theory_vs_simulation.py
```

Each `run_*.py` script writes its outputs to the corresponding `results/data/` directory. Model parameters are set in `sim_config.py` and recorded in each folder's `config.json`.
