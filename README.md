# BanditPy

Tools for loading, structuring, simulating, and fitting two-armed bandit behavior. The codebase now centers on a flexible `DecisionModel` + `Policy` stack, with pluggable optimizers and lightweight data containers.

## Highlights
- Data containers in [banditpy/core/mab.py](banditpy/core/mab.py) for trial-wise probabilities, choices, rewards, and session/block/window metadata, plus filtering, binning, and pandas export helpers.
- Turnkey loaders for hardware logs: `.dat` ingestion in [banditpy/io/datio.py](banditpy/io/datio.py) and CSV ingestion in [banditpy/io/csvio.py](banditpy/io/csvio.py) that reconstruct `Bandit2Arm` tasks with timing metadata.
- Policy-driven modeling in [banditpy/models/model.py](banditpy/models/model.py) with Q-learning, UCB, Thompson sampling, and state-inference policies under [banditpy/models/policy](banditpy/models/policy) and optimizer backends in [banditpy/models/optim.py](banditpy/models/optim.py).
- Posterior predictive checks and forward simulation via `DecisionModel.simulate_posterior_predictive()` and `DecisionModel.simulate_policy()` for synthetic datasets.
- Quick behavioral metrics such as switch probability in [banditpy/analyses/switch_probability.py](banditpy/analyses/switch_probability.py) and plotting helpers in [banditpy/plots](banditpy/plots).

## Installation
The repository is source-only (no packaging metadata). Add it to your Python path and install the scientific stack:

```pwsh
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip numpy pandas scipy joblib tqdm optuna matplotlib seaborn
setx PYTHONPATH "%PYTHONPATH%;C:\Users\asheshlab\Documents\Codes\BanditPy"
```

If you prefer editable installs, add a minimal `pyproject.toml`/`setup.py` first, then run `pip install -e .`.

## Quickstart

```python
from pathlib import Path

from banditpy.io.datio import dat2ArmIO
from banditpy.models.model import DecisionModel
from banditpy.models.policy import Qlearn2Arm, ThompsonShared2Arm
from banditpy.analyses.switch_probability import SwitchProb2Arm

# 1) Load raw logs → Bandit2Arm
task = dat2ArmIO(Path("/path/to/session_folder"))
task = task.filter_by_trials(min_trials=80, clip_max=400)

# 2) Fit a policy with multiple random starts
model = DecisionModel(task, policy=Qlearn2Arm(), reset_mode="session")
model.fit(n_starts=20, optimizer="de", n_jobs=4, progress=True)
print(model.params)

# 3) Compare an alternative policy (shared Thompson sampling)
th_model = DecisionModel(task, policy=ThompsonShared2Arm(use_analytic=True))
th_model.fit(n_starts=10, optimizer="lbfgs")
print(th_model.params)

# 4) Posterior predictive simulation
sim_task = model.simulate_posterior_predictive(seed=0)

# 5) Basic behavior metric
switch_rate = SwitchProb2Arm(task).by_session()
print(f"Mean switch probability: {switch_rate:.3f}")
```

## Policy Catalog (plug into `DecisionModel`)
- Q-learning: vanilla and perseverance-biased variants in [banditpy/models/policy/qlearn.py](banditpy/models/policy/qlearn.py)
- UCB family: empirical, RL-corrected, and Bayesian flavors in [banditpy/models/policy/ucb.py](banditpy/models/policy/ucb.py)
- Thompson sampling: shared vs. fully split learning rates, analytic or Monte Carlo logits in [banditpy/models/policy/thompson.py](banditpy/models/policy/thompson.py)
- State inference: latent state tracking with softmax decisions in [banditpy/models/policy/state_inference.py](banditpy/models/policy/state_inference.py)

Each policy exposes parameter bounds and defaults through `policy.bounds` and `policy.describe(as_markdown=True)` for quick reporting.

## Data Conventions
- Choices are 1/2; rewards are Bernoulli (0/1). `Bandit2Arm` normalizes probabilities given in 0–100% or 0–1 form.
- Session IDs should monotonically increase; helper methods auto-fix negative jumps and can infer blocks/windows from datetimes.
- `.dat` and `.csv` loaders expect the DIBA lab event schema (see loader docstrings) and preserve start/stop timestamps in milliseconds plus absolute `datetime` if provided.

## Fitting and Simulation Tips
- `reset_mode` controls when internal policy state resets: `"session"`, `"block"`, `"window"`, or a custom boolean mask matching the trial count.
- Optimizers: `lbfgs` (default), differential evolution (`"de"`), or Optuna (`"optuna"`). Configure seeds and bounds via the policy’s `bounds` registry.
- Use `DecisionModel.simulate_policy(...)` to generate synthetic datasets under a specified reward schedule and block structure.

## Contributing
- Open issues for new loaders, policies, or metrics you want to add.
- Keep docstrings current and prefer small reproducible examples for new features.
- Run lightweight fits on synthetic data before submitting changes to ensure likelihoods and gradients behave as expected.
