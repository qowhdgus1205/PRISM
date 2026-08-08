# PRISM

PRISM is a research codebase for **privileged intermediate-variable learning
under feasible inference constraints**. During training, the model can use
additional physical variables (ACP); at deployment, every feasible model must
predict the target from deployable inputs alone.

```text
deployable X  ->  physical intermediate ACP  ->  target Y
       |                   training only
       +-------------------------------------> feasible prediction
```

The public paper package contains the 13 datasets used in the manuscript's
Main, Case1, and Case2 evaluations. Every processed table has fewer than
100,000 rows.

## Repository layout

```text
prism/            reusable models, losses, training, and analysis package
scripts/*.py      dataset preparation, experiments, and diagnostics
scripts/*.sh      paper orchestration and repository checks
results/          generated reports and selected tracked paper artifacts
data/processed/   13 publication-ready tables (clean public release only)
```

[PAPER_DATASETS.md](PAPER_DATASETS.md) lists the exact paper groups, processed
table sizes, and upstream attribution.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU execution is optional. PyTorch automatically falls back to CPU when CUDA
is unavailable.

## Quick start

Run all Main, Case1, and Case2 experiments:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_paper_experiments.sh
```

The preparation step downloads the official UCI archives when raw files are
not already available. Raw archives are not included in the public release;
the bundled processed tables range from 103 to 34,168 rows.

For a short smoke run:

```bash
SEEDS="1" EPOCHS=2 PATIENCE=1 \
PROCESSED_MODELS="simple_mlp two_stage_mlp oracle_mlp prism" \
DEVICE=cpu bash scripts/run_paper_experiments.sh
```

To prepare only a subset, pass explicit dataset names to
`scripts/prepare_datasets.py --datasets ...`. Generated outputs are
written below `results/paper_cases/`.

## Role convention

Generic processed tables use a shared schema:

- `feature_*`: deployable inputs `X`
- `intermediate_*`: training-only privileged variables `ACP`
- `target_*`: prediction targets `Y`
- `ID_*`: sample or group identifiers used to prevent split leakage

Oracle models may use true ACP for reference. Models labeled feasible never
receive measured ACP at inference.

## Reproducibility and repository checks

```bash
bash scripts/check_repo.sh
```

This checks Python syntax, shell syntax, hard-coded workspace paths, accidental
secrets, and unignored files larger than GitHub's 100 MB object limit.

## Data and licensing

Large raw datasets, candidate-search data, checkpoints, and generated run
directories are excluded from version control. The clean public release
contains only the 13 processed paper tables and selected result summaries.
Dataset preparation scripts retain official source URLs and role metadata.
See `PAPER_DATASETS.md` for attribution and licensing.

PRISM source code is released under the [MIT License](LICENSE). Dataset files
remain subject to their upstream licenses and attribution requirements.

## Citation

The manuscript and citation metadata are being prepared. Until a formal
release is available, cite the repository commit used for an experiment.
