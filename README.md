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

- requirements.txt
- README.md

## Entry point: `smeef.py`

The main interactive/demo file is `smeef.py`. Use it to:

- Run a cinematic demo of the environment (uses `pygame` for visualization).
- Load a saved SB3 model (DQN/PPO/A2C) if available and drive the agent with it, otherwise run a random/exploratory demo.

Key sections inside `smeef.py` (high level structure):

- Constants & enhanced visual configuration (colors, palette, CELL_SIZE)
- Particle effects & small visual helper classes (Particle)
- Utility helpers (safe_float, safe_rect_args)
- Model loader: `load_model()` — attempts to load the best available model
- Rendering functions: `draw_grid()`, `draw_animated_agent()`, `draw_resource_section()`, etc.
- Demo runner: `run_demo()` — main loop that steps the env, queries the model (or samples random actions), renders, and prints terminal logs
- `if __name__ == '__main__':` — UX-friendly welcome banner and `run_demo()` launch

To run the interactive demo (recommended after installing requirements):

```powershell
python smeef.py
```

You can change which algorithm/model is used by editing the `ALGORITHM` and `MODEL_PATHS` constants near the top of `smeef.py`.


## Mapping to the assignment requirements

### 1) Custom environment

- Implemented in `environment/smeef_env.py`.
- Actions: discrete, mission-specific set implemented as an `Action` Enum (movement, USE_SERVICE, WORK_PART_TIME, ATTEND_TRAINING, SEEK_SUPPORT, etc.). The list is exhaustive and relevant to the scenario—see the file header for the full list.
- Observation space: a Dict with keys: `position` (2 ints), `resources` (4 floats), `needs` (4 floats), `child_status` (2 floats). For MLP policies we provide `NormalizeFlattenObs` that returns a 12-dim normalized Box.
- Rewards: shaped rewards composed of resource changes, need reductions, service bonuses and termination penalties. Each step's `info['reward_components']` gives a breakdown for analysis.
- Start state: implemented in `reset()` (home position, default resources/needs/child_status values).
- Terminal conditions: reaching the goal, critical resource depletion, or exceeding `max_steps`.

### 2) Visualization & static demo

- The env supports `render()` using `pygame` (lazy-imported). For the static demo (required by the assignment), use `run_random_demo.py` to step the environment with random actions and optionally save frames under `outputs/videos/`.

### 3) Algorithms implemented

- DQN (value-based): `training/dqn_training.py` (Stable-Baselines3)
- PPO (policy optimization): `training/ppo_demo.py` (SB3)
- A2C (actor-critic): `training/a2c_ultra_fast.py` (SB3)
- REINFORCE (vanilla policy gradient): `training/reinforce_vanilla.py` and `training/reinforce_training.py` (PyTorch)

All algorithms are intended to operate on the same environment (via `NormalizeFlattenObs`) for objective comparison. `compare_all.py` provides evaluation utilities and fallbacks for models saved with the original dict observations.

### 4) Hyperparameter sweeps

- `reinforce_training.py` contains a built-in 10-config grid (`--sweep`). For SB3 scripts, adapt the training launcher or wrap them with a simple sweep driver to run 10+ hyperparameter combinations. Save each run's metadata and final reward to `outputs/metrics/summary.csv`.

### 5) Recording the agent and video

- Create the required static demo (random actions) using `run_random_demo.py`.
- To record the best agent: run the trained model and capture your screen (assignment requires full-screen recording plus camera on). The `render()` method gives a GUI window for the environment. Save frames or a video under `outputs/videos/` and keep the terminal transcript (training/evaluation log).

### 6) Requirements and reproducibility

Install dependencies and create a venv:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The code uses Gymnasium-style reset/step signatures. If you have an older Gym installation, install `gymnasium` and prefer it for compatibility.

### 7) Submission artifacts

Include the following in your submission (GitHub repo named `student_name_rl_summative`):

- `models/` — best model checkpoints for each algorithm
- `outputs/metrics/summary.csv` — per-run hyperparameters and final metrics
- `outputs/plots/` — learning curves & `algorithm_comparison.png`
- `outputs/videos/` — random demo and best-agent screen recording
- A short PDF report (2–4 pages) summarizing the environment, reward structure, hyperparameter choices, results and short discussion (use repository figures)

## Usage examples

- Quick model comparison (loads models from `models/`):

```powershell
python training/compare_all.py
```

- Short REINFORCE smoke test (200 episodes):

```powershell
python training/reinforce_training.py --total-episodes 200
```

- Run REINFORCE sweep (10 configs, override episodes):

```powershell
python training/reinforce_training.py --sweep --total-episodes 1000
```

- Run the random visualization demo (static frames):

```powershell
python run_random_demo.py --save-frames outputs/videos/random_demo
```

## Grading & rubric alignment (quick checklist)

- Environment validity & complexity: document state/action/reward/termination clearly in the report.
- Policy training & performance: include average reward, steps/episode, convergence curves, and a table of best configs; save logs under `outputs/logs/`.
- Simulation visualization: ensure your video shows GUI and terminal logs as required.
- Stable Baselines / Policy Gradient implementation: provide hyperparameter justification and at least 10 different runs per algorithm.
- Discussion & analysis: include clear figures, concise captions, and an interpretation of why algorithms behaved as they did.

## Troubleshooting

- If you see dict vs Box observation shape errors during evaluation, use `NormalizeFlattenObs` when training/evaluating or keep/restore the VecNormalize wrapper used at training time.
- If `model.predict` or `env.step` throw indexing errors, print the observation/action types and convert action arrays to Python ints before calling `env.step` (this project includes safe-eval fallbacks in `compare_all.py`).

---

If you want, I can generate a short PDF report skeleton with placeholders for figures and text, and a sample `outputs/metrics/summary.csv` template you can fill during runs. Want me to create those now?


