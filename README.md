# Enhancing Fractional Gradient Descent with Learned Optimizers

Research code for **[Enhancing Fractional Gradient Descent with Learned Optimizers](https://arxiv.org/abs/2510.18783)** (J. Sobotka, P. Šimánek), which introduces **L2O-CFGD**: *Learning to Optimize Caputo Fractional Gradient Descent*.

Fractional gradient descent (FGD) replaces the integer-order gradient with a fractional-order (Caputo) derivative. This gives the update a tunable dependence on the optimization history, but it also adds hyperparameters (the fractional order `alpha`, the shift `beta`, and the integral terminal `c`) whose good values are problem-dependent and hard to schedule, especially in non-convex settings. L2O-CFGD meta-learns an LSTM that *emits those hyperparameters at every step*, so the white-box CFGD update stays intact while its schedule is learned from data. The result outperforms CFGD with statically searched hyperparameters, is competitive with a fully black-box learned optimizer, and, unlike the black-box one, leaves a schedule one can plot and interpret.

This repository contains the CFGD/FGD optimizers, the learned-optimizer (L2O) wrapper, the meta-training loop, the optimizees and tasks, and the notebooks that produce the paper's figures.

---

## Installation

```bash
conda env create -f environment.yml
conda activate fl2o
pip install -e .
```

`environment.yml` pins Python 3.10 with PyTorch (CUDA 11.8), torchvision, matplotlib, seaborn and `lovely-tensors`.

### Environment variables

Copy `.env.example` to `.env` and fill it in:

```bash
DATA_PATH="/path/to/data"          # torchvision data root (MNIST)
CKPT_PATH="/path/to/checkpoint/folder"
DEVICE="cpu"                        # or "cuda"
```

Nothing in the code loads `.env` itself. VS Code picks it up automatically for notebooks and debugging; from a shell, export it first:

```bash
set -a; source .env; set +a
```

`DATA_PATH` is read at import time in [fl2o/data.py](fl2o/data.py) and must be set before `import fl2o.data`. `DEVICE` and `CKPT_PATH` default gracefully (`"cpu"` / unused) but the notebooks expect all three. The MNIST dataset is loaded with `download=False`, so download it into `DATA_PATH` yourself once.

---

## Repository layout

```
fl2o/
├── optimizer.py          # FGD, AFOGD, CFGD (+ closed forms), GD, Adam, L2O_Update
├── l2o.py                # L2O — the LSTM that predicts base-optimizer hyperparameters
├── optimizee.py          # MLPOptee, CustomParams, MNISTConv
├── optimizee_modules.py  # MetaModule / MetaLinear / MetaConv2d - differentiable-through modules
├── data.py               # MNIST, H1/H2/H3, CustomTask, generate_least_squares_task
├── training.py           # do_fit (one optimization run), meta_train, learning-rate searches
└── utils.py              # plotting for the paper's figures, Gauss–Jacobi helper
scripts/                  # notebooks reproducing the paper's experiments
```

---

## Core abstractions

Everything is organized around three objects that are composed by `do_fit`:

| Object | What it is | Interface |
| --- | --- | --- |
| **task** (`data_cls`) | The problem being solved. `.sample()` returns a dict with `x`, `y` and a `loss_fn(y_hat)` closure. | `MNIST`, `H1`/`H2`/`H3`, `CustomTask`, `generate_least_squares_task` |
| **optimizee** (`optee`) | The model being optimized. Subclasses `MetaModule`, so its parameters are plain tensors that stay in the autograd graph across update steps (this is what makes meta-training through the optimizer possible). | `MLPOptee`, `CustomParams`, `MNISTConv` |
| **optimizer** (`opter`) | The update rule. `.step()` writes new parameters back onto the optimizee. | `GD`, `Adam`, `FGD`, `AFOGD`, `CFGD`, `CFGD_ClosedForm(_v2)`, `L2O` |

`do_fit(...)` runs one optimization trajectory and returns `(log, optee, opter)`, where `log` holds per-iteration `loss`, wall-clock `time`, and anything you register in `additional_metrics`. `meta_train(config)` runs `do_fit` repeatedly with `in_meta_training=True`, backpropagating `log(loss)` through the unrolled trajectory into the L2O weights every `unroll` steps, and keeps the best checkpoint by summed loss.

### Optimizers

| Class | Description |
| --- | --- |
| `GD`, `Adam` | Baselines, with the same `step(task, optee)` convention as the rest. |
| `FGD` | Fractional (stochastic) GD with higher-order truncation and fixed memory step `K` ([ref](https://arxiv.org/abs/1901.05294v2)). |
| `AFOGD` | Adaptive fractional-order (accelerated) GD ([ref](https://arxiv.org/pdf/2303.04328v1.pdf)). |
| `CFGD` | Caputo FGD ([ref](https://arxiv.org/abs/2104.02259)), `version="NA"` (non-adaptive terminal) or `"AT"` (adaptive terminal, needs `init_points`). Uses Gauss–Jacobi quadrature with `s` sample points and a Hutchinson estimator for `diag(H)`. |
| `CFGD_ClosedForm`, `CFGD_ClosedForm_v2` | Closed-form CFGD for quadratic objectives. `v2` takes `alpha`/`beta` separately, the original takes the combined `gamma = beta - (1 - alpha)/(2 - alpha)`. |
| `L2O_Update` | Placeholder base optimizer that applies whatever update the LSTM emits (fully black-box L2O baseline). |
| `L2O` | The learned optimizer itself: an LSTM wrapped around one of the base optimizers above, predicting the entries of `params_to_optimize` at every step. |

### Learning rates

`lr` may be a float **or a callable**, which lets a run use an exact or searched step size instead of a tuned constant:

- `get_optimal_lr` - exact optimal step size for a quadratic model, used by the least-squares and quadratic experiments.
- `parallel_n_step_lookahead_lr_search_hfunc_tanh_twolayer_optee` / `n_step_lookahead_lr_search_...` / `per_param_...` - one-step lookahead line searches used by the H-function experiments. As their names say, these are specialized to the two-layer tanh optimizee.
- `find_best_lr(...)` - an offline grid search over full runs, for the baselines.

---

## Quickstart

### Run a hand-tuned optimizer

```python
import torch, torch.nn as nn
from fl2o.data import H2
from fl2o.optimizee import MLPOptee
from fl2o.optimizer import CFGD
from fl2o.training import (
    do_fit,
    parallel_n_step_lookahead_lr_search_hfunc_tanh_twolayer_optee as lr_search,
)

log, optee, opter = do_fit(
    opter_cls=CFGD,
    opter_config={
        "lr": lr_search,          # or a float
        "alpha": 0.95, "beta": 0.0, "c": torch.tensor([-5.0]),
        "s": 1, "n_hutchinson_steps": 5, "version": "NA",
    },
    optee_cls=MLPOptee,
    optee_config={
        "layer_sizes": [50], "inp_size": 1, "out_size": 1,
        "act_fn": nn.Tanh(), "output_bias": False,
    },
    data_cls=H2,
    data_config={"preload_n_samples": 100},
    n_iters=50,
)
print(log["loss"][0], "->", log["loss"][-1])
```

### Meta-train L2O-CFGD

```python
import torch.nn as nn, torch.optim as optim
from fl2o.data import H2
from fl2o.optimizee import MLPOptee
from fl2o.optimizer import CFGD
from fl2o.l2o import L2O
from fl2o.training import (
    meta_train, do_fit,
    parallel_n_step_lookahead_lr_search_hfunc_tanh_twolayer_optee as lr_search,
)

config = {
    "data": {"data_cls": H2, "data_config": {"preload_n_samples": 100}},
    "optee": {
        "optee_cls": MLPOptee,
        "optee_config": {
            "layer_sizes": [50], "inp_size": 1, "out_size": 1,
            "act_fn": nn.Tanh(), "output_bias": False,
        },
    },
    "opter": {
        "opter_cls": L2O,
        "opter_config": {
            "in_dim": 3,                                  # len(in_features) + 1
            "out_dim": 3,                                 # alpha, beta, c
            "hidden_sz": 40,
            "in_features": ("grad", "iter_num_enc"),
            "base_opter_cls": CFGD,
            "base_opter_config": {
                "lr": lr_search, "alpha": None, "beta": None, "c": None,
                "s": 1, "version": "NA", "init_points": None,
                "detach_gauss_jacobi": True,
            },
            "params_to_optimize": {
                "alpha": {"idx": 0, "act_fns": ("sigmoid",)},
                "beta":  {"idx": 1, "act_fns": ("identity",)},
                "c":     {"idx": 2, "act_fns": ("identity",)},
            },
        },
    },
    "meta_training_config": {
        "meta_opter_cls": optim.Adam,
        "meta_opter_config": {"lr": 3e-4},
        "n_runs": 1200,      # meta-training trajectories
        "unroll": 40,        # truncated BPTT length
        "loggers": None,
    },
    "n_iters": 600,          # optimizee steps per trajectory
    "additional_metrics": None,
    "ckpt_config": None,     # or {"ckpt_every_nth_run": 20, "ckpt_dir_meta_training": ...}
}

l2o_dict, l2o_dict_best, log = meta_train(config=config)
```

### Meta-test the learned optimizer

Pass the trained `l2o_dict` back into `do_fit` — the opter class/config are then ignored:

```python
log, optee, opter = do_fit(
    opter_cls=None, opter_config=None,
    optee_cls=config["optee"]["optee_cls"],
    optee_config=config["optee"]["optee_config"],
    data_cls=config["data"]["data_cls"],
    data_config=config["data"]["data_config"],
    n_iters=5000,                                 # longer than meta-training: tests generalization
    l2o_dict=l2o_dict_best["best_l2o_dict"],      # or l2o_dict for the last checkpoint
)
```

`do_fit` prints a warning and forces `unroll = 1` outside of meta-training; that is expected.

---

## Experiments

The notebooks in [scripts/](scripts/) cover the paper's experiments: meta-training, meta-testing against baselines, and the strategy analysis.

| Notebook | Problem |
| --- | --- |
| [quadratic_fig1.ipynb](scripts/quadratic_fig1.ipynb) | 2-D quadratic `f(x, y) = 10x² + y²` (Figure 1), `CFGD_ClosedForm` vs. GD vs. L2O-CFGD |
| [least_squares.ipynb](scripts/least_squares.ipynb) | Least squares `min ½‖Wᵀx − y‖²` with `d = m = 100`, `CFGD_ClosedForm_v2` base |
| [h_funcs.ipynb](scripts/h_funcs.ipynb) | Univariate tanh two-layer network fitting H1/H2/H3 (non-convex), `CFGD` base |
| [mnist.ipynb](scripts/mnist.ipynb) | MNIST classification with a small MLP optimizee |

Each notebook follows the same shape: build `config` → `meta_train` → define a `runs` dict of baselines (GD, Adam, NA-CFGD and AT-CFGD hyperparameter sweeps, black-box L2O) → run them all through `do_fit` with a shared seed → compare with `plot_metrics` and dissect the schedule with `plot_strategy`. Meta-testing runs are cached under `CKPT_PATH` and reloaded on re-run unless you set `"load_saved": False` on a run.

One caveat for a fresh clone: the two cells right after the imports load previously meta-trained checkpoints from `CKPT_PATH` (e.g. `l2o/01-08_12-50__L2O__CFGD/ckpt_1200.pt`), and those checkpoints are not in the repository. Skip those cells and meta-train from scratch, or point them at your own checkpoints.

---

## Citation

```bibtex
@article{sobotka2025enhancing,
  title   = {Enhancing Fractional Gradient Descent with Learned Optimizers},
  author  = {Sobotka, Jan and {\v{S}}im{\'a}nek, Petr},
  journal = {arXiv preprint arXiv:2510.18783},
  year    = {2025}
}
```
