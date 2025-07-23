#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_eval.sh – batch evaluator for mesh results
# ------------------------------------------------------------------------------
# * Searches every first‑level directory inside $EXP_DIR (default ./exp_results)
#   that contains *.obj meshes (econ, gt_aligned, etc.)
# * Executes `evaluate_models.py --input_dir <subject_dir>`
#   where evaluate_models.py is the Python script you pasted above.
# * Collects each subject’s `evaluation_results.csv` into one master CSV and
#   prints a per‑model average table at the end.
#
#   Usage examples
#       ./run_eval.sh                                   # defaults
#       EXP_DIR=./exp_results/thuman ./run_eval.sh      # custom root
# ------------------------------------------------------------------------------
set -euo pipefail

# ---------------- configuration ---------------------------------------------
EXP_DIR="${EXP_DIR:-./exp_results}"
PY_SCRIPT="${PY_SCRIPT:-evaluate_models.py}"   # path to the evaluator
CSV_MERGED="${CSV_MERGED:-batch_evaluation.csv}"
# -----------------------------------------------------------------------------

shopt -s nullglob
dir_array=("${EXP_DIR}"/*/)
if (( ${#dir_array[@]} == 0 )); then
  echo "No subject folders found under $EXP_DIR" >&2
  exit 1
fi

echo "Found ${#dir_array[@]} subject folders – starting evaluation…"

# prepare merged CSV header later
rm -f "$CSV_MERGED"
header_written=false

for DIR in "${dir_array[@]}"; do
  subj="$(basename "$DIR")"
  echo -e "\n================  $subj  ================"

  python "$PY_SCRIPT" --input_dir "$DIR" 2>&1 | tee "$DIR/evaluate.log"
  status=${PIPESTATUS[0]}
  if [ $status -ne 0 ]; then
    echo "✘ $subj evaluation failed – see $DIR/evaluate.log" >&2
    continue
  fi

  csv_path="$DIR/evaluation_results.csv"
  if [ ! -f "$csv_path" ]; then
    echo "⚠ $subj produced no evaluation_results.csv" >&2
    continue
  fi

  # append to merged CSV (write header only once)
  if ! $header_written; then
    cat "$csv_path" > "$CSV_MERGED"
    header_written=true
  else
    tail -n +2 "$csv_path" >> "$CSV_MERGED"
  fi

done

if ! $header_written; then
  echo "No evaluation CSVs were collected – exiting." >&2
  exit 1
fi

# --------------------- summary table ----------------------------------------
python - <<'PY'
import pandas as pd, sys, pathlib
csv=sys.argv[1]
df=pd.read_csv(csv)
print("\n================  Aggregate summary  ================")
print("Averaged per model across subjects:\n")
print(df.groupby('model').mean(numeric_only=True))
PY "$CSV_MERGED"

echo -e "\nMerged CSV written to $CSV_MERGED"
