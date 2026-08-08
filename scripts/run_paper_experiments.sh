#!/usr/bin/env bash
# Reproduce the PRISM Main, Case1, and Case2 dataset experiments.

set -euo pipefail

SEEDS="${SEEDS:-1 2 3 4 5}"
EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LATENT_DIM="${LATENT_DIM:-128}"
LR="${LR:-5e-4}"
DEVICE="${DEVICE:-cuda}"
PREPARE_DATA="${PREPARE_DATA:-auto}"
MODELS="${PROCESSED_MODELS:-simple_mlp random_forest xgboost lightgbm cnn_mlp ftt_mlp two_stage_mlp oracle_mlp prism prism_z_random_forest prism_z_xgboost prism_z_lightgbm prism_z_acp_random_forest prism_z_acp_xgboost prism_z_acp_lightgbm}"

MAIN_DATASETS="airfoil_self_noise energy_efficiency hydraulic_systems household_power_consumption concrete_slump_test"
CASE1_DATASETS="wine_quality superconductivity auto_mpg combined_cycle_power_plant"
CASE2_DATASETS="air_quality parkinsons_telemonitoring real_estate_valuation student_performance"
PAPER_DATASETS="${MAIN_DATASETS} ${CASE1_DATASETS} ${CASE2_DATASETS}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/paper_cases}"

cd "${REPO_ROOT}"

needs_prepare=0
for dataset in ${PAPER_DATASETS}; do
  if [ ! -f "${REPO_ROOT}/data/processed/${dataset}/${dataset}_prism.csv" ]; then
    needs_prepare=1
    break
  fi
done
if [ "${PREPARE_DATA}" = "1" ] || { [ "${PREPARE_DATA}" = "auto" ] && [ "${needs_prepare}" = "1" ]; }; then
  python -u scripts/prepare_datasets.py --datasets ${PAPER_DATASETS}
fi

python -u scripts/run_experiments.py \
  --datasets ${PAPER_DATASETS} \
  --models ${MODELS} \
  --seeds ${SEEDS} \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --latent-dim "${LATENT_DIM}" \
  --lr "${LR}" \
  --full-acp-lambda 0.5 \
  --distill-lambda 0.05 \
  --contrastive-lambda 0.05 \
  --mixup-lambda 0.10 \
  --mixup-alpha 0.4 \
  --mixup-k 5 \
  --pretrain-acp-epochs "${EPOCHS}" \
  --pretrain-acp-patience "${PATIENCE}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}"

python -u scripts/run_acp_diagnostic.py \
  --datasets ${PAPER_DATASETS} \
  --models random_forest ridge \
  --primary-model random_forest \
  --seeds ${SEEDS} \
  --rf-estimators 100 \
  --n-jobs -1 \
  --target-results "${OUTPUT_DIR}/processed_extra_results.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --no-progress

echo "Paper experiments complete: ${OUTPUT_DIR}"
