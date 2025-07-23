#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_mexecon.sh – batch‑run ECON on every subject folder
# ------------------------------------------------------------------------------
# 1.  Finds every first‑level directory inside $EXP_DIR (defaults to ./experiments).
# 2.  Calls the Python module you specify (default *apps.infer_exp*, can be
#     changed with MODULE=apps.infer_icon).
# 3.  Times each run and keeps the console log.
# 4.  Reads the *_smpl_optimization_metrics.csv* produced by the run, averages
#     **all rows (every Stage + View)** per metric, and collects the numbers.
# 5.  Prints a per‑subject table plus a grand average at the end.
#
# Usage examples (from repo root)
#   ./run_mexecon.sh                            # use defaults
#   MODULE=apps.infer_icon EXP_DIR=./examples/thuman ./run_mexecon.sh
#   GPU_ID=1 FRONT_VIEW=0 BACK_VIEW=4 ./run_mexecon.sh
# ------------------------------------------------------------------------------

set -euo pipefail

# ------------------------- configurable bits ---------------------------------
CONFIG="${CONFIG:-./configs/econ.yaml}"
EXP_DIR="${EXP_DIR:-./experiments}"
OUT_ROOT="${OUT_ROOT:-./exp_results}"
FRONT_VIEW="${FRONT_VIEW:-0}"
BACK_VIEW="${BACK_VIEW:-1}"
GPU_ID="${GPU_ID:-0}"
MODULE="${MODULE:-apps.infer_exp}"
# -----------------------------------------------------------------------------

shopt -s nullglob  # if no match, glob expands to empty array instead of literal
SUBJECT_DIRS=("${EXP_DIR}"/*/)

if (( ${#SUBJECT_DIRS[@]} == 0 )); then
  echo "No subject directories found under $EXP_DIR – exiting." >&2
  exit 1
fi

echo "Found ${#SUBJECT_DIRS[@]} subjects in $EXP_DIR"

TMP_METRIC_FILE="$(mktemp)"
echo "Subject,Time_s,Silhouette_IoU,Landmark_Error_px,Normal_Loss,Head_Roll_Loss,Total_Loss" > "$TMP_METRIC_FILE"

for SUBJECT_DIR in "${SUBJECT_DIRS[@]}"; do
  SUBJECT="$(basename "$SUBJECT_DIR")"
  echo -e "\n================  $SUBJECT  ================"
  OUT_DIR="$OUT_ROOT/$SUBJECT"
  mkdir -p "$OUT_DIR"

  # ---------------- run ECON --------------------------------------------------
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
    continue
  fi

  # -------------- locate metrics CSV -----------------------------------------
  # look for either <id>_smpl_optimization_metrics.csv *or* smpl_optimization_metrics.csv
  CSV_PATH=$(find "$OUT_DIR" -type f \( -name "*_smpl_optimization_metrics.csv" -o -name "smpl_optimization_metrics.csv" \) | head -n1)
  if [ -z "$CSV_PATH" ]; then
    echo "⚠ No metrics CSV produced for $SUBJECT – skipping" >&2
    continue
  fi

    # -------------- average ALL rows/Stages ------------------------------------
  metrics=$(python - "$CSV_PATH" <<'PY'
import pandas as pd, sys
csv_path = sys.argv[1]
df = pd.read_csv(csv_path, index_col=[0,1])
mean = df.mean()  # averages across every Stage & View
print(mean['Silhouette IoU ↑'], mean['Landmark Error (px) ↓'], mean['Normal Loss ↓'], mean['Head Roll Loss ↓'], mean['Total Loss ↓'])
PY
  )

  # split the five numbers into shell vars
  read -r SIL LMK NORM ROLL TOTAL <<< "$metrics"

  # -------------- append row -------------------------------------------------- --------------------------------------------------
  echo "$SUBJECT,$ELAPSED,$SIL,$LMK,$NORM,$ROLL,$TOTAL" >> "$TMP_METRIC_FILE"

done

# ----------------------- print nice table ------------------------------------
if command -v column >/dev/null; then
  printf "\nPer‑subject summary:\n" && column -s, -t "$TMP_METRIC_FILE"
else
  cat "$TMP_METRIC_FILE"
fi

# ----------------------- grand average ---------------------------------------
python - <<'PY'
import pandas as pd, sys, os
csv = sys.argv[1]
df = pd.read_csv(csv)
print("\nGrand average (numeric columns):")
print(df.mean(numeric_only=True))
PY "$TMP_METRIC_FILE"

# ----------------------- clean up -------------------------------------------
rm "$TMP_METRIC_FILE"
