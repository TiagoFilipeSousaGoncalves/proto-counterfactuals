# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A study on counterfactual generation using **ProtoPNet** and **Deformable ProtoPNet** (with baseline CNNs as comparison). The codebase trains, tests, and analyzes prototype-based models on three medical/biological image datasets — **PH2** (dermoscopy), **PAPILA** (retinal fundus), and **CUB2002011** (birds) — then generates counterfactual explanations and label-coherence metrics. The companion paper lives under `paper/`.

## Environment

The project uses **pixi** for dependency management. All commands should be run through pixi (e.g., `pixi run python ...`).

- `pixi.toml` defines the `protocounterfactuals` environment. PyTorch is installed from the CUDA 12.4 index on linux/win and the CPU index on osx-arm64 (see `[target.osx.pypi-dependencies]`).
- Python `>=3.11,<3.13`. Dependencies include `torch`, `torchvision`, `scikit-learn`, `wandb`, `opencv-contrib-python-headless`, `scikit-image`, `openpyxl`, `torchinfo`, `pandas`, `tqdm`, `matplotlib`, `augmentor`.
- Logging goes to Weights & Biases — `wandb.login()` is called at the top of training scripts. Configure your `WANDB_API_KEY` env var before running.

## High-level architecture

The repo is split into a **library** under `src/` and **SLURM job scripts** under `scripts/`. Each model family has its own self-contained package, and each dataset has its own preprocessing subpackage.

```
src/
├── baseline/                  # DenseNet/ResNet/VGG baselines (no prototypes)
├── protopnet/                 # Original ProtoPNet (Chen et al.)
├── deformable-protopnet/      # Deformable-ProtoPNet
│   └── Deformable-Convolution-V2-PyTorch/   # vendored DCNv2 ops
├── data_preprocessing/        # per-dataset split scripts (uses StratifiedShuffleSplit)
└── source_ppnet_repos/        # vendored upstream reference repos (read-only history)
```

Inside each model package the modules follow a consistent naming convention:

| File pattern                         | Purpose |
| ------------------------------------ | ------- |
| `models_train.py`                    | training loop (CLI entry point) |
| `models_test.py`                     | evaluation on train/val/test splits |
| `models_inference_and_prototypes.py` | inference pass + prototype bookkeeping |
| `models_local_analysis.py`           | per-image local explanations |
| `models_global_analysis.py`          | dataset-level analysis |
| `models_prototype_pruning.py`        | prototype pruning |
| `models_counterfactuals_*.py`        | counterfactual retrieval / explanations / label-coherence |
| `prototypes_*.py`                    | prototype utilities (push, retrieval, stats, label-coherence) |
| `data_utilities.py`                  | dataset classes (`CUB2002011Dataset`, `PAPILADataset`, `PH2Dataset`, `preprocess_input_function`) |
| `model_{vgg,resnet,densenet}_utilities.py` | backbones |
| `model_utilities.py`                 | `PPNet` / `construct_PPNet` and baseline wrappers |
| `train_val_test_utilities.py`        | `joint`, `warm_only`, `last_only`, `model_train`, `model_validation`, `print_metrics` |
| `receptive_field_utilities.py`       | receptive-field bookkeeping (proto layer only) |

## Key conventions (mirrored from `AGENTS.md` and the source)

- **Train/val/test splits** are produced by the per-dataset `src/data_preprocessing/<dataset>/split_train_test.py` using `StratifiedShuffleSplit` across `n_folds` (default 5). `train_size + val_size + test_size` must equal 1.0.
- **Training returns a metrics dict** containing at least `loss_train`, `loss_val`, and `run_avg_loss` (see `src/protopnet/models_train.py:525`).
- **Model testing** is performed by `model_test(...)` and expects train, val, and test loaders (see `src/baseline/models_test.py:228`).
- **Debugging image-related issues** is usually done by loading with `matplotlib.pyplot.imread` (see `src/deformable-protopnet/data_utilities.py:99`).
- **Trust executable code over docs** when there's a conflict.
- **Pipeline order** for a full run: `[Data Prep] → [Train] → [Test] → [Inference & Prototypes] → [Local/Global Analysis] → [Counterfactuals] → [Label Coherence]`.
- **Script standardization**: all training/analysis shell scripts under `scripts/<family>/<dataset>/` follow the same template — SLURM directives at the top, `echo` start/finish banners, and a single `python` invocation. When adding or modifying one, match the format used by the other `ph2/`, `papila/`, `cub2002011/` scripts in the same family (see `AGENTS.md` for the rationale).

## Running the pipeline

### 1. Install / activate environment

```bash
pixi install
pixi shell                       # or prefix every command with `pixi run`
```

### 2. Prepare a dataset (example: PH2)

```bash
python src/data_preprocessing/ph2/split_train_test.py \
  --data_dir /path/to/ph2-database \
  --seed 42 \
  --train_size 0.70 --val_size 0.10 --test_size 0.20 \
  --n_folds 5
```

The same pattern applies to `data_preprocessing/papila/` and `data_preprocessing/cub2002011/`. PAPILA also has a `utilities.py` for dataset-specific helpers; CUB uses its standard splits.

### 3. Train a model

All training scripts are SLURM batch files. Adapt the data dir (and optionally SLURM directives) to your environment, then either submit via `sbatch` or run the inner `python` command directly.

ProtoPNet on PH2 with ResNet34 (from `scripts/protopnet/ph2/ph2_train_resnet34.sh`):

```bash
python src/protopnet/models_train.py \
  --data_dir /path/to/ph2-database \
  --dataset ph2 \
  --base_architecture resnet34 \
  --batchsize 32 \
  --num_workers 4 \
  --gpu_id 0 \
  --folds 0 1 2 3 4 \
  --output_dir results
```

Each model family supports `--base_architecture` ∈ `{densenet121, densenet161, resnet34, resnet50, resnet152, vgg16, vgg19}`. The Deformable package additionally accepts `resnet50`. Use `--dataset` ∈ `{cub2002011, papila, ph2, STANFORDCARS}`.

Common useful flags (see `src/protopnet/models_train.py` for the canonical list):
- `--folds 0 1 2 3 4` — which k-fold splits to train (must be a contiguous or explicit subset).
- `--timestamp YYYY-MM-DD_HH-MM-SS` — pin a run id; otherwise one is auto-generated.
- `--resume --checkpoint <path>` — resume from a saved checkpoint.
- `--classweights` — enable class-weighted CE loss for imbalanced datasets.
- `--coefs 'crs_ent,clst,sep,l1'` — override loss coefficients; default `{'crs_ent': 1, 'clst': 0.8, 'sep': -0.08, 'l1': 1e-4}`.
- `--push_start <N>` — epoch at which prototype pushing begins.
- `--num_train_epochs`, `--num_warm_epochs`, `--last_layer_optimizer_lr`.

### 4. Test / analyze

Each step has a matching shell script in `scripts/<family>/<dataset>/`. The naming convention is `<dataset>_<step>.sh` (e.g. `ph2_test.sh`, `ph2_inference_and_prototypes.sh`, `ph2_local_analysis.sh`, `ph2_counterfactual_retrieval.sh`, `ph2_counterfactual_label_coherence.sh`, `ph2_prototype_label_coherence.sh`, `ph2_prototypes_images_stats.sh`, `ph2_prototypes_inference_stats.sh`). Each script is a list of `python` invocations for every backbone in `{densenet121, resnet34, vgg16}` — comment/uncomment lines as needed.

The baseline family only has `*_train_*.sh`, `*_test.sh`, `*_counterfactual_label_coherence.sh`, and `*_counterfactual_retrieval.sh` (no prototype-specific steps).

The Deformable family additionally ships `compute_counterfactual_label_coherence.sh` in each dataset folder.

`scripts/cub2002011/` contains older global-analysis and pruning scripts that target the legacy `code/` path; they are kept for reference but the active workflows are under `scripts/protopnet/cub2002011/`.

### 5. Counterfactuals

Two scripts in the same family produce counterfactuals:
- `models_counterfactuals_retrieval.py` — nearest-counterfactual search in `--feature_space` ∈ `{conv_features, proto_features}`.
- `models_counterfactuals_explanations.py` — visual/saliency explanation of retrieved counterfactuals.
- `models_counterfactuals_labels_coherence.py` — inter-model label agreement using `--coherence_metric` (e.g. `fleiss_kappa`).

## Where to make common changes

| You want to…                                  | Look at / edit |
| --------------------------------------------- | -------------- |
| Add a new dataset                             | `src/data_preprocessing/<new>/split_train_test.py` + a `Dataset` class in each model family's `data_utilities.py` + a `if dataset == ...` branch in `models_train.py` |
| Add a new backbone                            | `model_{arch}_utilities.py` per family, then add it to `base_architecture_to_features` in `model_utilities.py` and to the `choices=` list in the `argparse` of every CLI entry point |
| Add a new counterfactual metric               | one of `models_counterfactuals_*.py` modules |
| Tweak loss coefficients or schedule           | `src/protopnet/models_train.py` (and the matching deformable version) — `coefs`, `push_start`, `num_warm_epochs` |
| Add a new analysis step                       | drop a `<step>.py` in the model package, then a shell script under `scripts/<family>/<dataset>/` following the SLURM template |

## Notes

- The vendored `src/source_ppnet_repos/ProtoPNet-master/` and `Deformable-ProtoPNet-main/` are upstream reference code; do not edit them — mirror changes into the active `src/protopnet/` and `src/deformable-protopnet/` packages instead.
- `src/deformable-protopnet/Deformable-Convolution-V2-PyTorch/` is a vendored DCNv2 implementation. It must be built (`sh make.sh` inside that folder) before deformable training will run.
- `data/`, `results/`, `wandb/`, `.pixi/`, and `*.{err,out}` are gitignored. Local datasets and trained checkpoints are not committed.
- There are no automated tests in this repo. Validation is done by running the train/test scripts end-to-end and inspecting metrics.
