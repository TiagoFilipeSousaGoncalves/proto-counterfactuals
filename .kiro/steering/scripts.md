---
inclusion: fileMatch
fileMatchPattern: "scripts/**"
---

# Shell Script Conventions

All scripts under `scripts/` are SLURM batch jobs. When adding or modifying any script, follow the exact template used by the existing ones.

## Template structure

```bash
#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --mem=<mem>
#SBATCH --job-name=<dataset>_<arch_abbrev>   # Job name
#SBATCH -o <dataset>_<step>.out              # STDOUT
#SBATCH -e <dataset>_<step>.err              # STDERR

echo "<DATASET> | Started | <Step>"

echo "<Family> | <Architecture>"
python src/<family>/models_<step>.py \
 --data_dir '/path/to/dataset' \
 --dataset <dataset> \
 --base_architecture <arch> \
 --batchsize <N> \
 --num_workers <N> \
 --gpu_id 0 \
 ...
 # --timestamp ''

echo "<DATASET> | Finished | <Step>"
```

## Rules

- Always include `#!/bin/bash` and `#SBATCH` directives at the top.
- Always wrap the body with `echo "... | Started | ..."` and `echo "... | Finished | ..."` banners.
- Test/analysis scripts list one `python` invocation per backbone (`densenet121`, `resnet34`, `vgg16` by default) — comment/uncomment lines as needed rather than deleting them.
- Training scripts typically have one invocation per file (one backbone per script).
- Leave `# --timestamp ''` commented out as a convenience placeholder.
- Match the naming pattern: `<dataset>_<step>.sh` (e.g. `ph2_test.sh`, `papila_counterfactual_retrieval.sh`).

## Directory layout per family

```
scripts/
├── baseline/
│   └── <dataset>/   # *_train_*.sh, *_test.sh, *_counterfactual_*.sh
├── protopnet/
│   └── <dataset>/   # full set: train, test, inference, local/global analysis,
│                    #   counterfactual retrieval/explanations/label_coherence,
│                    #   prototype stats/label_coherence
└── deformable-protopnet/
    └── <dataset>/   # same as protopnet + compute_counterfactual_label_coherence.sh
```

Baseline scripts do **not** have prototype-specific steps (no `inference_and_prototypes`, `local_analysis`, `global_analysis`, `prototype_*`).
