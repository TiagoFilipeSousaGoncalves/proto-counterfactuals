# Environment & Commands

## Dependency manager

This project uses **pixi**. All Python commands must be prefixed with `pixi run` (or run inside `pixi shell`).

```bash
pixi install        # install / sync the environment
pixi shell          # activate the environment interactively
pixi run python ... # run a single command without activating
```

## Python version

`>=3.11, <3.13` (defined in `pixi.toml`)

## PyTorch platform notes

- **Linux / Windows**: PyTorch installed from the CUDA 12.4 index (`https://download.pytorch.org/whl/cu124`).
- **macOS (osx-arm64)**: PyTorch installed from the CPU index (`https://download.pytorch.org/whl/cpu`). CUDA is not available on this platform.

## Key dependencies (from `pixi.toml`)

| Package | Version |
|---|---|
| torch | >=2.5.1 |
| torchvision | >=0.20.1 |
| scikit-learn | >=1.7.2 |
| wandb | >=0.21.3 |
| opencv-contrib-python-headless | >=4.11.0.86 |
| scikit-image | >=0.25.2 |
| torchinfo | >=1.8.0 |
| pandas | >=2.3.2 |
| matplotlib | >=3.10.8 |
| augmentor | >=0.2.12 |

## Weights & Biases

Training scripts call `wandb.login()` at startup. Set `WANDB_API_KEY` in your environment before running any training.

## Data preparation (example: PH2)

```bash
pixi run python src/data_preprocessing/ph2/split_train_test.py \
  --data_dir /path/to/ph2-database \
  --seed 42 \
  --train_size 0.70 --val_size 0.10 --test_size 0.20 \
  --n_folds 5
```

Same pattern for `papila/` and `cub2002011/`.

## Training (example: ProtoPNet on PH2)

```bash
pixi run python src/protopnet/models_train.py \
  --data_dir /path/to/ph2-database \
  --dataset ph2 \
  --base_architecture resnet34 \
  --batchsize 32 \
  --num_workers 4 \
  --gpu_id 0 \
  --folds 0 1 2 3 4 \
  --output_dir results
```

Supported `--base_architecture` values: `densenet121`, `densenet161`, `resnet34`, `resnet50`, `resnet152`, `vgg16`, `vgg19`.  
Supported `--dataset` values: `cub2002011`, `papila`, `ph2`, `STANFORDCARS`.

Useful training flags:
- `--folds 0 1 2 3 4` — which k-fold splits to train
- `--timestamp YYYY-MM-DD_HH-MM-SS` — pin a run ID
- `--resume --checkpoint <path>` — resume from checkpoint
- `--classweights` — class-weighted CE loss for imbalanced datasets
- `--coefs 'crs_ent,clst,sep,l1'` — override loss coefficients (defaults: `crs_ent=1, clst=0.8, sep=-0.08, l1=1e-4`)
- `--push_start <N>` — epoch at which prototype pushing begins
- `--num_train_epochs`, `--num_warm_epochs`, `--last_layer_optimizer_lr`

## Deformable ProtoPNet — one-time build step

```bash
cd src/deformable-protopnet/Deformable-Convolution-V2-PyTorch
sh make.sh
```

Must be done before any deformable training run.
