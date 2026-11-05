# Experiment recipes

Run commands from the repository root. Follow the [README](../README.md) to
install dependencies and download the backbone, checkpoints, and datasets.
The examples below use TabICLv2 and a CUDA device.

BO and BED commands enable the MLP's remaining-budget feature by default;
AL leaves it disabled by default. Both AL head types support the opt-in setting
`train.policy_head.use_remaining_budget=true`; the target-aware head appends the
budget to its FFN inputs. Evaluation restores the saved setting and scale.
See [remaining acquisition budget](../README.md#remaining-acquisition-budget)
for details and legacy-checkpoint compatibility.

All three trainers use immediate reward credit by default (`train.loss.gamma=0`).
Set `train.loss.gamma=1` for undiscounted reward-to-go, or a value between 0 and 1
for discounted reward-to-go. In older BO and AL code, `gamma=1` applied equal
time weights to immediate rewards; use `gamma=0` here to preserve that training
behavior. Existing checkpoints can still be evaluated without changing their
saved training configuration.

## Loss-driven active learning

Train on 1D GP functions with two initial observations, 200 candidates,
100 targets, and 20 acquisitions. The defaults use a one-layer target-aware
Transformer head, weights proportional to `exp(10 * y)`, batch size 50,
10,000 updates, and learning rate `1e-5` for both the head and backbone:

```bash
uv run python scripts/train_al_policy.py seed=41 device=cuda output.run_tag=al
```

Evaluate on 100 GP episodes:

```bash
uv run python scripts/evaluate_al.py \
  +policy.checkpoints.polar=checkpoints/policy/al_gp_regression/tabicl/al/checkpoint.pt \
  seed=42 device=cuda
```

Evaluate on Forrester, Gramacy--Lee, Higdon, and Sine--Gaussian Bumps,
using 100 episodes per function (400 trajectories in total):

```bash
uv run python scripts/evaluate_al.py task=benchmark_functions \
  +policy.checkpoints.polar=checkpoints/policy/al_gp_regression/tabicl/al/checkpoint.pt \
  seed=42 device=cuda
```

Both commands compare POLAR with GP-RS, GP-US, GP-VR, and GP-VR-w using
20 acquisitions and `exp(10 * y)` evaluation weights. Table A2 reports
`weighted_target_rmse_end` and the negative of `weighted_target_log_prob_end`
in `metrics.json`, with their corresponding standard errors.

To evaluate an existing paper checkpoint, replace the checkpoint path with
`notebooks/final_results/loss_driven_al/polar/checkpoint.pt`.
Set `train.finetune_epochs=0` for a frozen backbone, or
`train.loss.policy_target_weighting=uniform` for ordinary AL.

## HPO-B

Use the following training settings. All spaces use batch size 50, five
initial observations, 50 acquisitions, and seed 42. Set
`train.finetune_epochs` equal to `train.num_epochs`.

| Search space | Epochs | Candidates | Targets |
| --- | ---: | ---: | ---: |
| 4796 | 200 | 200 | 50 |
| 5859 | 100 | 200 | 50 |
| 5860 | 100 | 70 | 10 |
| 5889 | 50 | 70 | 10 |
| 5906 | 100 | 50 | 10 |
| 5527 | 1000 | 1000 | 100 |

Train on search space 4796:

```bash
uv run python scripts/train_bo_policy.py \
  task=hpob task.search_space_id=4796 \
  task.n_query=200 task.n_target=50 \
  train.num_epochs=200 train.finetune_epochs=200 \
  train.batch_size=50 train.num_acquisitions=50 \
  seed=42 device=cuda output.run_tag=hpob_4796
```

For other spaces, replace the search-space ID, epochs, candidates, and targets
using the table above.

Evaluate on the official test tasks and initialisations:

```bash
uv run python scripts/evaluate_bo.py \
  task=hpob task.search_space_id=4796 \
  compare.methods='[polar,tabicl_ei_mc,gp_ei,random]' \
  +policy.checkpoints.tabicl=notebooks/final_results/hpob/4796/polar/checkpoint.pt \
  eval.num_acquisitions=50 eval.query_pool_size=0 \
  eval.ei_mc_samples=32 eval.metric=normalized_regret \
  seed=42 device=cuda
```

Key the checkpoint by `tabicl` so POLAR and the TabICL EI baseline use the same
fine-tuned backbone. Replace both `task.search_space_id` and the checkpoint
path when evaluating another space.

## Dockstring

Prepare the fingerprint cache as described in the README, then train with the
1000-epoch configuration:

```bash
uv run python scripts/train_bo_policy.py \
  --config-name exp/train_dockstring_policy device=cuda
```

Evaluate on the held-out docking targets:

```bash
uv run python scripts/evaluate_bo.py \
  task=dockstring task.split=test \
  compare.methods='[polar,tabicl_ei_mc,rf_ei,gp_ei,random]' \
  +policy.checkpoints.tabicl=notebooks/final_results/dockstring/polar/checkpoint.pt \
  eval.num_acquisitions=50 eval.query_pool_size=2000 \
  models.gp.kernel=tanimoto eval.max_query_chunk=256 \
  seed=42 device=cuda
```

For the RBF GP baseline, use `compare.methods='[gp_ei]'` and
`models.gp.kernel=rbf`.

## Bayesian experimental design

### Training

Location finding, 2D:

```bash
uv run python scripts/train_bed_policy.py \
  task=location_finding task.dim_x=2 \
  task.n_query=1000 task.n_target=20 task.n_contrastive=99999 \
  train.num_epochs=1200 train.finetune_epochs=1200 \
  train.batch_size=50 train.num_acquisitions=30 \
  train.optim.lr_policy=5e-3 train.optim.lr_backbone=1e-4 \
  seed=42 device=cuda
```

Location finding, 5D:

```bash
uv run python scripts/train_bed_policy.py \
  task=location_finding task.dim_x=5 \
  task.n_query=1000 task.n_target=20 task.n_contrastive=99999 \
  train.num_epochs=1430 train.finetune_epochs=1430 \
  train.batch_size=50 train.num_acquisitions=50 \
  train.optim.lr_policy=5e-3 train.optim.lr_backbone=1e-4 \
  seed=42 device=cuda
```

CES:

```bash
uv run python scripts/train_bed_policy.py \
  task=ces task.n_query=2000 task.n_target=10 task.n_contrastive=99999 \
  train.num_epochs=25000 train.finetune_epochs=25000 \
  train.batch_size=20 train.num_acquisitions=10 \
  train.optim.lr_policy=3e-4 train.optim.lr_backbone=1e-4 \
  seed=42 device=cuda
```

### Evaluation

Location finding, 2D:

```bash
uv run python scripts/evaluate_bed.py \
  task=location_finding task.dim_x=2 \
  compare.methods='[polar,random]' \
  +policy.checkpoints.polar=notebooks/final_results/location_finding/2D_2K/polar/checkpoint_likelihood_evals_3000000.pt \
  eval.num_acquisitions=30 eval.n_query_eval=2000 device=cuda
```

Location finding, 5D:

```bash
uv run python scripts/evaluate_bed.py \
  task=location_finding task.dim_x=5 \
  compare.methods='[polar,random]' \
  +policy.checkpoints.polar=notebooks/final_results/location_finding/5D_2K/polar/checkpoint_likelihood_evals_5001500.pt \
  eval.num_acquisitions=50 eval.n_query_eval=2000 device=cuda
```

CES:

```bash
uv run python scripts/evaluate_bed.py \
  task=ces compare.methods='[polar,random]' \
  +policy.checkpoints.polar=notebooks/final_results/ces/polar/checkpoint_likelihood_evals_10000000.pt \
  eval.num_acquisitions=10 eval.n_query_eval=20000 \
  eval.n_contrastive_eval=10000000 device=cuda
```

Evaluation uses 1000 episodes, with 1,000,000 contrastive samples for location
finding and 10,000,000 for CES.
For a quick check, add `eval.n_eval_episodes=8 eval.n_contrastive_eval=128`.

## Plotting

Download the results into `notebooks/final_results/`, then run the notebooks
under `notebooks/`. For cells using `apply_style(use_tex=True)`, install a TeX
distribution or set `use_tex=False`.

## Outputs

- Training checkpoints: `checkpoints/policy/<family>_<task>/<backbone>/<run_tag>/checkpoint.pt`
- Training metrics: `results/<family>_<task>/<backbone>/<run_tag>/`
- Evaluation results: `results/evaluate_<family>_<task>/<run_tag>/`

Set `output.run_tag=<name>` to name a run.
