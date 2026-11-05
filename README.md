# POLAR

Official implementation of [Efficient Adaptive Data Acquisition via Pretrained
Belief Representations](https://arxiv.org/abs/2606.25197).

[Paper](https://arxiv.org/abs/2606.25197) ·
[Checkpoints and results](https://huggingface.co/club-amortization/POLAR) ·
[Experiment recipes](docs/experiment_recipes.md)

POLAR turns pretrained tabular
predictors into amortized policies for adaptive data acquisition. A lightweight
policy head scores candidate queries using candidate-conditioned TabICL or
TabPFN embeddings; the policy is trained with a task-specific utility for
active learning (AL), Bayesian optimization (BO), or Bayesian experimental
design (BED).

This repository contains POLAR training, evaluation, plotting, and checkpoint
loading code for the paper's main experiments.

## What is included

- One Python package, `polar`, under `src/polar`.
- TabICLv2 and TabPFN regression backbones with local-only checkpoint loading.
- POLAR training for AL, BO, and BED.
- Reproducible task implementations for synthetic regression, HPO-B,
  Dockstring, location finding, and CES.
- Random, Gaussian-process, random-forest, and TFM acquisition baselines.
- Hydra configurations, tests, and paper plotting notebooks.

The paper's primary experiments use the TabICLv2 regression backbone. TabPFN
support is retained for ablations and extensions.

## Installation

Python 3.10 or newer and a recent PyTorch installation are required.

```bash
git clone https://github.com/huangdaolang/POLAR.git
cd POLAR
uv sync --locked
```

All training and evaluation commands load model weights from local paths. A
fresh clone already contains the expected directory skeleton, so download the
TabICLv2 regression checkpoint into the folder that is waiting for it:

```text
checkpoints/pretrained/tabicl/v2/tabicl-regressor-v2-20260212.ckpt
```

TabPFN weights, if you want the ablations, go in
`checkpoints/pretrained/tabpfn/v2/` and `checkpoints/pretrained/tabpfn/v2.5/`.

Download the original backbone separately; it is not part of the POLAR result
bundle:

```bash
uvx --from huggingface_hub hf download jingang/TabICL \
  tabicl-regressor-v2-20260212.ckpt \
  --local-dir checkpoints/pretrained/tabicl/v2
```

## Datasets

The BED tasks (location finding, CES) and the AL tasks (GP prior,
benchmark functions) are synthetic and need no download. The two BO
benchmarks read local files under `data/`, which is not tracked in Git.

**HPO-B.** Download `hpob-data.zip` from the official
[HPO-B repository](https://github.com/releaunifreiburg/HPO-B) (the archive is
hosted at
`https://github.com/sebastianpinedaar/hpo-data/raw/refs/heads/main/hpob-data.zip`)
and unpack it so that the JSON files sit directly under `data/hpob/`:

```text
data/hpob/
  meta-train-dataset.json
  meta-validation-dataset.json
  meta-test-dataset.json
  bo-initializations.json
```

`task.data_root` (default `data/hpob`) points at this folder, and
`task.search_space_id` selects the search space. The paper uses 4796, 5859,
5860, 5889, 5906, and 5527.

**DOCKSTRING.** Download `dockstring-dataset.tsv` from the
[DOCKSTRING Figshare record](https://figshare.com/articles/dataset/dockstring_dataset/16511577)
and place it at `data/dockstring/dockstring-dataset.tsv` (`task.raw_data_path`).
The first Dockstring run subsamples 10,000 molecules and writes the 512-bit
Morgan-fingerprint cache to
`data/dockstring/cache/random10k_seed42_morgan512.pt` (`task.cache_path`).
Building the cache needs RDKit (`uv sync --extra chem`); once the cache
exists, the raw TSV and RDKit are no longer required.

Validate the environment before running experiments:

```bash
uv run ruff check .
uv run pytest -q
```

## Quickstart

Run commands from the repository root. Hydra writes run outputs under
`results/` and policy checkpoints under `checkpoints/policy/`. These commands
exercise the pipelines; use the task-specific settings in the
[experiment recipes](docs/experiment_recipes.md) to reproduce reported runs.

Train POLAR:

```bash
# Loss-driven active learning
uv run python scripts/train_al_policy.py device=cuda

# HPO-B Bayesian optimization
uv run python scripts/train_bo_policy.py \
  task=hpob task.search_space_id=4796 device=cuda

# Location-finding Bayesian experimental design
uv run python scripts/train_bed_policy.py \
  task=location_finding device=cuda
```

Evaluate a released checkpoint with the canonical method name `polar`:

```bash
# HPO-B: POLAR vs. the paper baselines.
# Key the checkpoint by BACKBONE (`tabicl`), not by method: POLAR and the
# TabICL baseline must share the same finetuned backbone, as in the paper.
uv run python scripts/evaluate_bo.py \
  task=hpob task.search_space_id=4796 \
  compare.methods='[polar,tabicl_ei_mc,gp_ei,random]' \
  +policy.checkpoints.tabicl=notebooks/final_results/hpob/4796/polar/checkpoint.pt \
  device=cuda

# Location finding: POLAR vs. random
uv run python scripts/evaluate_bed.py \
  task=location_finding task.dim_x=2 compare.methods='[polar,random]' \
  +policy.checkpoints.polar=notebooks/final_results/location_finding/2D_2K/polar/checkpoint_likelihood_evals_3000000.pt \
  device=cuda

# Active learning: use a locally trained AL checkpoint
uv run python scripts/evaluate_al.py \
  +policy.checkpoints.polar=/path/to/al/checkpoint.pt \
  device=cuda
```

### Remaining acquisition budget

Policy heads can condition on the number of acquisitions left (including the
current one), divided by
`train.policy_head.budget_scale`. This scale defaults to
`train.num_acquisitions` and is restored from the checkpoint at evaluation,
even when the evaluation horizon changes. The remaining count is capped by
the available candidate pool.

MLP heads append the scalar to each candidate embedding. Target-aware heads
append it to each FFN input after cross-attention: the FFN input grows from
`d` to `d + 1`, while its output, attention, and residual dimensions remain `d`.
For the default AL head, this changes `512 -> 1024 -> 512` to
`513 -> 1024 -> 512` and adds 1024 parameters. Backbone inputs and embedding
dimensions stay the same.

Budget conditioning is enabled by default for BO and BED and disabled by
default for AL, for either head type. To enable it for AL, append
`train.policy_head.use_remaining_budget=true` to the training command.
Use `train.policy_head.use_remaining_budget=false` to disable it.

Evaluation infers the setting and scale from the checkpoint. Legacy checkpoints
load with their original architecture, including early target-aware checkpoints
whose recorded budget flag was ignored: their FFN weights identify the actual
input width. New budget weights start at zero, preserving the original
same-seed initialization. Old state dicts can also warm-start an explicitly
enabled budget head by zero-padding the relevant input layers; use a fresh
optimizer because those layers change shape.
The effect of budget conditioning must be learned during training.

### Method names

AL, BO, and BED evaluation share one grammar, defined in
[`src/polar/utils/method_names.py`](src/polar/utils/method_names.py). The
*prefix* selects a backbone and the *suffix* labels one of several checkpoints
that share a backbone:

| Name | Backbone | Mode |
| --- | --- | --- |
| `polar` | `policy.backend` | policy |
| `polar_<label>` | `policy.backend` | policy |
| `<backend>_polar` | the backend in the name | policy |
| `<backend>_polar_<label>` | the backend in the name | policy |
| `<backend>_<acquisition>` | the backend in the name | surrogate |
| `random` | none | random |

`policy.backend` (default `tabicl`) is the single knob that pairs a backbone
with the bare `polar` forms, and it is spelled the same way in all three
scripts. Use `polar` in public commands and reports; use `<backend>_polar` when
one run must compare several backbones, for example the TabICL-vs-TabPFN
backbone ablation:

```bash
compare.methods='[tabicl_polar,tabpfn_polar]' \
+policy.checkpoints.tabicl=/path/to/tabicl.pt \
+policy.checkpoints.tabpfn=/path/to/tabpfn.pt
```

Two script-specific notes: BED has no surrogate baselines, so
`<backend>_<acquisition>` does not apply there; and AL has no standalone
`random`, because every AL method also needs a backbone to fit and score the
target set — use `<backend>_random`, e.g. `gp_random`. The pre-release
`<backend>_policy` spelling is rejected with an error that lists the accepted
names.

Checkpoints are looked up by method name first, then by backbone name, then
by `policy.default_checkpoint_path`. Keying by backbone is what lets `polar`
and `tabicl_ei_mc` share one finetuned backbone. If a surrogate method finds
no checkpoint it silently falls back to the *pretrained* backbone, which is
not the paper's setup — check the `surrogate_backbone_state` line in the run
log.

See [`docs/experiment_recipes.md`](docs/experiment_recipes.md) for training and
evaluation commands and task-specific settings.

## Checkpoints and results

Checkpoints and plotting inputs are hosted publicly at
[club-amortization/POLAR](https://huggingface.co/club-amortization/POLAR).
Large artifacts are excluded from Git. Download the bundle from the repository
root (approximately 4.6 GB):

```bash
uvx --from huggingface_hub hf download club-amortization/POLAR \
  --include 'final_results/*' --local-dir notebooks
```

The remote folder is named `final_results`, so `--local-dir notebooks` restores
the `notebooks/final_results/` paths expected by the plotting notebooks. To
download just one checkpoint instead:

```bash
uvx --from huggingface_hub hf download club-amortization/POLAR \
  final_results/hpob/4796/polar/checkpoint.pt --local-dir notebooks
```

The current bundle includes POLAR checkpoints for all six HPO-B search spaces,
Dockstring, 2D and 5D location finding, and CES, plus BED ablations and saved
comparison results.

```text
checkpoints/
  pretrained/   # original TabICL / TabPFN weights, as downloaded
  policy/       # new POLAR training outputs, one directory per run

notebooks/final_results/   # downloaded paper checkpoints, metrics, plotting inputs
```

New training runs write to
`checkpoints/policy/<family>_<task>/<backbone>/<run_tag>/checkpoint.pt`.
Evaluation can load a checkpoint from either this directory or the downloaded
`notebooks/final_results/` tree by passing its path explicitly.

POLAR checkpoint payloads load with `weights_only=True`. Fine-tuned runs store
both the policy head and the adapted backbone; frozen-backbone runs may store
only the head. Reconstructing either variant still requires the corresponding
original backbone checkpoint locally.

### Publishing a checkpoint bundle

Training checkpoints can carry substantial AdamW optimizer state that is not
needed for inference. Collect the runs you want to publish into a
staging directory, then export a clean copy without touching the originals:

```bash
uv run python scripts/export_release_checkpoints.py \
  --source-root /path/to/staged-checkpoints \
  --output-root /path/to/polar-checkpoints
```

The exporter drops optimizer, scheduler, and AMP state, clears machine-specific
paths, preserves the directory structure, and writes `manifest.json` with
source and release SHA-256 hashes. Ship `LICENSE`, `THIRD_PARTY_NOTICES.md`,
and `licenses/` alongside it when distributing a new bundle. The existing
Hugging Face bundle contains the original experiment artifacts; downloading it
does not run this exporter or remove training state.

## Reproducibility and paper alignment

- Random seeds are set centrally; evaluation episodes are shared across
  compared methods.
- Checkpoints are loaded with PyTorch's restricted `weights_only=True` mode.
- Plotting notebooks read relative paths from the repository root.

## Repository map

```text
config/      Hydra experiment, model, task, and training configuration
notebooks/   Paper analysis and plotting notebooks
scripts/     Training, evaluation, preparation, and release utilities
src/polar/   Library code
tests/       Unit and integration tests
docs/        Experiment recipes
```

## Licensing

POLAR's original code is released under the MIT License. Vendored TabICL code
is BSD-3-Clause licensed. Vendored TabPFN code is governed by the Prior Labs
License, and TabPFN model weights have their own terms. Read
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the files under
[`licenses/`](licenses/) before redistributing code or checkpoints.

Built with PriorLabs-TabPFN.

This attribution is required by section 10 of the Prior Labs License v1.2,
which applies because this repository redistributes TabPFN source. The same
section requires that any distributed model derived from TabPFN source,
weights, or outputs carry a name beginning with "TabPFN" — this applies to the
TabPFN-backed ablation checkpoints, not to the TabICL-backed POLAR
checkpoints that produce the paper's main results.

## Citation

If you use POLAR in your research, please cite:

```bibtex
@article{huang2026efficient,
  title={Efficient Adaptive Data Acquisition via Pretrained Belief Representations},
  author={Huang, Daolang and Huang, Zhuoyue and Hassan, Conor and Acerbi, Luigi and Kaski, Samuel and Rainforth, Tom},
  journal={arXiv preprint arXiv:2606.25197},
  year={2026}
}
```
