# Project: proto-counterfactuals

## What this repo is

A study on counterfactual generation using **ProtoPNet** and **Deformable ProtoPNet** (with baseline CNNs as comparison). The codebase trains, tests, and analyzes prototype-based models on three datasets:

- **PH2** — dermoscopy images
- **PAPILA** — retinal fundus images
- **CUB2002011** — bird species images

It then generates counterfactual explanations and label-coherence metrics. The companion paper lives under `paper/`.

## Architecture overview

```
src/
├── baseline/               # DenseNet/ResNet/VGG baselines (no prototypes)
├── protopnet/              # Original ProtoPNet (Chen et al.)
├── deformable-protopnet/   # Deformable-ProtoPNet (includes vendored DCNv2 ops)
├── data_preprocessing/     # Per-dataset split scripts (ph2/, papila/, cub2002011/)
└── source_ppnet_repos/     # Vendored upstream reference repos — do NOT edit
```

Scripts (SLURM batch jobs) live under `scripts/<family>/<dataset>/`.

## Module naming conventions

Each model package (`protopnet/`, `deformable-protopnet/`, `baseline/`) follows this pattern:

| File | Purpose |
|---|---|
| `models_train.py` | Training loop (CLI entry point) |
| `models_test.py` | Evaluation on train/val/test splits |
| `models_inference_and_prototypes.py` | Inference pass + prototype bookkeeping |
| `models_local_analysis.py` | Per-image local explanations |
| `models_global_analysis.py` | Dataset-level analysis |
| `models_prototype_pruning.py` | Prototype pruning |
| `models_counterfactuals_*.py` | Counterfactual retrieval / explanations / label-coherence |
| `prototypes_*.py` | Prototype utilities (push, retrieval, stats, label-coherence) |
| `data_utilities.py` | Dataset classes + `preprocess_input_function` |
| `model_{vgg,resnet,densenet}_utilities.py` | Backbone definitions |
| `model_utilities.py` | `PPNet` / `construct_PPNet` and baseline wrappers |
| `train_val_test_utilities.py` | `joint`, `warm_only`, `last_only`, `model_train`, `model_validation`, `print_metrics` |
| `receptive_field_utilities.py` | Receptive-field bookkeeping (proto layer only) |

## Key conventions

- **Train/val/test splits** use `StratifiedShuffleSplit` via `src/data_preprocessing/<dataset>/split_train_test.py`. `train_size + val_size + test_size` must equal exactly `1.0`.
- **Training metrics dict** must contain at least `loss_train`, `loss_val`, and `run_avg_loss` (see `src/protopnet/models_train.py:525`).
- **Model testing** calls `model_test(...)` with train, val, and test loaders (see `src/baseline/models_test.py:228`).
- **Debugging images** uses `matplotlib.pyplot.imread` (see `src/deformable-protopnet/data_utilities.py:99`).
- **Trust executable code over docs** when there's a conflict.

## Full pipeline order

```
[Data Prep] → [Train] → [Test] → [Inference & Prototypes]
  → [Local/Global Analysis] → [Counterfactuals] → [Label Coherence]
```

## Common changes guide

| Goal | Where to look / edit |
|---|---|
| Add a new dataset | `src/data_preprocessing/<new>/split_train_test.py` + `Dataset` class in each family's `data_utilities.py` + branch in `models_train.py` |
| Add a new backbone | `model_{arch}_utilities.py` per family → `base_architecture_to_features` in `model_utilities.py` → `choices=` in every CLI argparse |
| Add a new counterfactual metric | `models_counterfactuals_*.py` |
| Tweak loss coefficients/schedule | `src/protopnet/models_train.py` (and the deformable equivalent) |
| Add a new analysis step | New `<step>.py` in the model package + shell script under `scripts/<family>/<dataset>/` |

## Notes

- `src/source_ppnet_repos/` is read-only upstream reference code. Mirror any changes into the active `src/protopnet/` and `src/deformable-protopnet/` packages.
- `src/deformable-protopnet/Deformable-Convolution-V2-PyTorch/` must be built (`sh make.sh` inside that folder) before deformable training will run.
- `data/`, `results/`, `wandb/`, `.pixi/`, and `*.{err,out}` are gitignored.
- There are no automated tests. Validation is done by running train/test scripts end-to-end and inspecting metrics.
