# SMEEF_RL — Assignment-ready project

This repository contains a custom mission-based environment (SMEEF) and training/evaluation scripts for four RL methods required by the assignment: Value-based (DQN) and Policy-based (REINFORCE, PPO, A2C). The primary demo/entry point for interactive visualization and playing back saved models is `smeef.py` (see below). The code and documentation are organized to make it straightforward to run experiments, repeat hyperparameter sweeps (10+ runs per algorithm), produce recordings, and compile a short report.

## Repository layout

project_root/

- environment/
	- `smeef_env.py`        # Custom Gymnasium environment implementation (dict observation)
	- `obs_wrappers.py`     # `NormalizeFlattenObs` to flatten dict -> 12-D Box for MLP policies

- agents/
	- `reinforce_agent.py`  # PyTorch policy network used by the vanilla REINFORCE trainer

- training/
	- `dqn_training.py`     # DQN (SB3) training script
	- `a2c_ultra_fast.py`   # A2C (SB3) quick runner
	- `ppo_demo.py`         # PPO (SB3) example runner
	- `reinforce_vanilla.py`# Vanilla PyTorch REINFORCE implementation
	- `reinforce_training.py` # Sweep-capable runner for REINFORCE (10-config grid)
	- `compare_all.py`      # Loads saved models, evaluates, and creates a comparison plot

- models/
	- `dqn/`, `ppo/`, `a2c/`, `reinforce/`  # Saved model artifacts

- outputs/
	- `plots/`, `metrics/`, `videos/`, `logs/` # Generated experiment outputs

# SMEEF_RL — Assignment-ready Reinforcement Learning Project

An assignment-ready reinforcement learning project featuring a custom mission-based environment (SMEEF) and implementations of four RL algorithms: DQN, PPO, A2C and REINFORCE. This repository includes interactive visualization, static demos, training scripts, hyperparameter sweep tooling, saved models, and plotting utilities for evaluation and submission.

## Quick Start

Prerequisites

- Python 3.10+ (virtual environment recommended)
- Windows PowerShell examples shown below

Installation & setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Run interactive demo

```powershell
python smeef.py
```

Run static demo (save frames / video)

```powershell
python run_random_demo.py --save-frames outputs/videos/random_demo
```

## Repository structure

Top-level layout (important files and folders):

```
SMEEF_RL/
│
├── README.md
├── requirements.txt
├── smeef.py                  # Main interactive demo (pygame visualization)
│
├── environment/
│   ├── __init__.py
	├── smeef_env.py          # Custom Gymnasium environment
	└── obs_wrappers.py       # NormalizeFlattenObs (Dict → Box)
│
├── agents/
│   ├── __init__.py
│   └── reinforce_agent.py    # PyTorch policy used by REINFORCE
│
├── training/
│   ├── __init__.py
	├── dqn_training.py       # DQN (SB3) training script
	├── ppo_demo.py           # PPO example runner (SB3)
	├── a2c_ultra_fast.py     # A2C minimal fast script
	├── reinforce_vanilla.py  # Vanilla REINFORCE implementation
	├── reinforce_training.py # REINFORCE sweeps, 10+ configs
	└── compare_all.py        # Evaluation + comparison plot
│
├── models/
│   ├── dqn/                 # Saved DQN models
│   ├── ppo/                 # Saved PPO models
│   ├── a2c/                 # Saved A2C models
# SMEEF_RL

A reinforcement-learning project that implements a custom mission-based environment (SMEEF) plus training and demo utilities for several RL algorithms (DQN, PPO, A2C, REINFORCE). The repository contains the environment code, agent implementations, training scripts, saved models, plotting utilities and example demos/visualizations.

This README documents: quick setup, how to run demos and training scripts, where artifacts are stored, and a short repo map.

## Quick start

Requirements

- Python 3.10–3.12 (recommended)
- Create and activate a virtual environment before installing dependencies.

Windows PowerShell example:

```powershell
python -m venv .venv311
.venv311\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Notes

- There is a local virtual environment directory `.venv311` in this workspace — consider adding it to `.gitignore` if you push this repo.
- The project depends on packages listed in `requirements.txt`.

## Quick examples

- Run the interactive demo (pygame visualization):

```powershell
python smeef.py
```

- Run static / headless demo (saves frames or video):

```powershell
python run_random_demo.py --save-frames outputs/videos/random_demo
```

- Run a PPO example/demo:

```powershell
python ppo_demo.py
```

## Training scripts

Top training scripts live in the `training/` folder. Examples:

- `training/dqn_training.py` — DQN training (SB3)
- `training/ppo_training.py` — PPO training (SB3)
- `training/a2c_training.py` — A2C training (SB3)
- `training/reinforce_training.py` — REINFORCE (PyTorch) and sweep tooling
- `training/compare_all.py` — evaluate multiple saved models and produce comparison plots

Run (example):

```powershell
python training/dqn_training.py
python training/reinforce_training.py --total-episodes 500
python training/compare_all.py
```

See `config/training_config.yaml` for default training hyperparameters.

## Project layout (important files)

## Project structure (as requested)

Below is the repository structure produced from your PowerShell Get-ChildItem listing. Use this as the authoritative project layout.

- Top-level folders
	- `agents/`
	- `config/`
	- `environment/`
	- `models/`
	- `outputs/`
	- `scripts/`
	- `training/`

- Important top-level files (examples from workspace)
	- `create_diagram.py`
	- `enhanced_demo.py`
	- `ppo_demo.py`
	- `README.md` (this file)
	- `requirements.txt`
	- `results_dashboard.py`
	- `run_random_demo.py`
	- `smeef.py`
	- `smeef_demo.py`
	- `__init__.py`

- Local virtual environment (present in workspace)
	- `.venv311/`  (recommended to add to `.gitignore`)

- Notable subfolders & outputs in workspace
	- `models/a2c/`, `models/dqn/`, `models/ppo/`, `models/reinforce/`
	- `models/dqn/run_*` (per-run folders)
	- `environment/` contains `smeef_env.py`, `obs_wrappers.py`, `rendering.py`
	- `outputs/logs/`, `outputs/metrics/`, `outputs/plots/`, `outputs/videos/`
	- `outputs/logs/<algorithm>/` (e.g. `a2c`, `dqn`, `ppo`, `reinforce`)
	- `scripts/` contains plotting/analysis scripts: `generate_analysis_plots.py`, `plot_training_stability.py`, etc.
	- `training/` contains training runners: `a2c_training.py`, `dqn_training.py`, `ppo_training.py`, `reinforce_training.py`, `compare_all.py`

Notes

- I used your Get-ChildItem output as the canonical structure. If you want me to reorganize files into a different folder layout (move files on disk and update imports), I can do that — tell me the target structure and I'll perform a safe refactor and run quick checks.
- I also recommend adding `.venv311/`, `__pycache__/` and `*.pyc` to `.gitignore` to keep the repository clean; I'll add that next unless you'd rather manage `.gitignore` yourself.

## Models & outputs

- Trained models are stored under `models/<algorithm>/`. The repo contains several saved runs (zip/pt files).
- Experiment artifacts (metrics, plots, TensorBoard logs, videos) are under `outputs/` (e.g. `outputs/metrics/`, `outputs/plots/`, `outputs/videos/`).

## Environment summary

The SMEEF environment (`environment/smeef_env.py`) exposes a mission-based grid-like task. Observations are provided as a dict; for training with standard MLP policies use the wrapper in `environment/obs_wrappers.py` to flatten/normalize the observation into a Box.

Reward and termination logic are implemented in `smeef_env.py`. Use the `info` dict returned on each step for diagnostics (reward components, mission status, etc.).

## Usage notes & troubleshooting

- Observation-shape mismatch: ensure you apply the same `NormalizeFlattenObs` wrapper at training and inference.
- Model compatibility: SB3 models are `.zip` files; PyTorch policies are `.pt`/`.pth` files. Check `smeef.py` for the `MODEL_PATHS` constants to point playback to a specific file.
- Missing packages: install via `pip install -r requirements.txt` into the activated venv.

## Suggested submission checklist

If preparing this repository for a submission or external sharing, include:

1. Source code (all `.py` files under repo root, `environment/`, `agents/`, `training/`)
2. Trained model checkpoints under `models/` for the algorithms you want to demonstrate
3. `outputs/` with sample `plots/`, `metrics/summary.csv`, and `videos/`
4. A short report in `report/` containing environment description, reward shaping decisions, hyperparameter choices and key figures

## Next steps I can help with

- Shorten or expand this README
- Add a CONTRIBUTING.md, LICENSE, or .gitignore that excludes local venvs
- Create a one-page PDF report template in `report/`
- Add automated scripts to export `report/figures/` from `outputs/plots/`

---


