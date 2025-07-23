#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_econ.sh – minimal batch runner for ECON
# ------------------------------------------------------------------------------
# * Finds every first‑level directory under $EXP_DIR (defaults to ./experiments)
# * Calls the ECON module you specify (default apps.infer_exp) with the usual
#   flags.  **No metric parsing** – it just logs and records timing.
# * Prints a per‑subject runtime table and an average at the end.
#
# Usage examples
#   ./run_econ.sh                                   # default settings
#   MODULE=apps.infer_icon GPU_ID=1 ./run_econ.sh   # switch module / GPU
#   EXP_DIR=./examples/thuman OUT_ROOT=./exp_results/thuman ./run_econ.sh
# ------------------------------------------------------------------------------
set -euo pipefail

# ------------------------- configurable bits ---------------------------------
CONFIG="${CONFIG:-./configs/econ.yaml}"
EXP_DIR="${EXP_DIR:-./experiments}"
OUT_ROOT="${OUT_ROOT:-./exp_results}"
FRONT_VIEW="${FRONT_VIEW:-6}"
BACK_VIEW="${BACK_VIEW:-2}"
GPU_ID="${GPU_ID:-0}"
MODULE="${MODULE:-apps.infer_exp}"
# -----------------------------------------------------------------------------

shopt -s nullglob
SUBJECT_DIRS=("${EXP_DIR}"/*/)
if (( ${#SUBJECT_DIRS[@]} == 0 )); then
  echo "No subject directories found under $EXP_DIR – exiting." >&2
  exit 1
fi

echo "Found ${#SUBJECT_DIRS[@]} subjects in $EXP_DIR"

declare -a TIMES_SEC

echo -e "\nRunning ECON for each subject…"
for SUBJECT_DIR in "${SUBJECT_DIRS[@]}"; do
  SUBJECT="$(basename "$SUBJECT_DIR")"
  echo -e "\n================  $SUBJECT  ================"
  OUT_DIR="$OUT_ROOT/$SUBJECT"
  mkdir -p "$OUT_DIR"

  START=$(date +%s)
  python -m "$MODULE" \
    -cfg        "$CONFIG" \
    -in_dir     "$SUBJECT_DIR" \
    -out_dir    "$OUT_DIR" \
    -novis \
    -front_view "$FRONT_VIEW" \
    -back_view  "$BACK_VIEW" \
    -gpu "$GPU_ID" \
    2>&1 | tee "$OUT_DIR/infer.log"
  STATUS=${PIPESTATUS[0]}
  END=$(date +%s)
  ELAPSED=$(( END - START ))

  if [ $STATUS -ne 0 ]; then
    echo "✘ $SUBJECT failed – see $OUT_DIR/infer.log" >&2
  else
    TIMES_SEC+=("$ELAPSED:$SUBJECT")
    echo "✔ $SUBJECT finished in $ELAPSED s"
  fi

done

# ----------------------- runtime summary -------------------------------------
if (( ${#TIMES_SEC[@]} > 0 )); then
  printf "\nPer‑subject runtimes:\n"
  for entry in "${TIMES_SEC[@]}"; do
    IFS=: read -r sec subj <<<"$entry"
    printf "%s\t%s\n" "$subj" "$sec"
  done | column -t

  # grand average
  total=0
  for entry in "${TIMES_SEC[@]}"; do
    IFS=: read -r sec _ <<<"$entry"; total=$(( total + sec )); done
  avg=$(awk "BEGIN{printf \"%.1f\", $total/${#TIMES_SEC[@]}}")
  echo -e "\nAverage runtime: $avg s across ${#TIMES_SEC[@]} subjects"
fi
