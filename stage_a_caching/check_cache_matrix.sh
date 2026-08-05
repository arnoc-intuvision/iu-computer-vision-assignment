#!/bin/bash
set -u

CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
COUNTER=/workspace/npz_count.py
CORRUPTIONS=(beamsreducing missingcamera motionblur)
SEVERITIES=(1 2 3)
TRAIN_EXPECT=5715
VAL_EXPECT=1353

pass=0; fail=0; missing=0

check() {
  local f=$1 expect=$2 label=$3
  if [ ! -f "$f" ]; then
    printf "  %-34s %-10s MISSING (%s)\n" "$label" "---" "$(basename "$f")"
    missing=$((missing + 1)); return
  fi
  local n size
  n=$(python "$COUNTER" "$f" 2>/dev/null)
  size=$(du -h "$f" | cut -f1)
  if [ "$n" = "$expect" ]; then
    printf "  %-34s %-10s OK    (%s)\n" "$label" "$n/$expect" "$size"
    pass=$((pass + 1))
  else
    printf "  %-34s %-10s BAD RECORD COUNT (%s)\n" "$label" "$n/$expect" "$size"
    fail=$((fail + 1))
  fi
}

echo "======================================================"
echo "Stage A cache matrix check"
echo "cache dir: $CACHE_DIR"
echo "======================================================"
echo ""
echo "TRAIN split (expect $TRAIN_EXPECT records each)"
echo "------------------------------------------------------"
check "$CACHE_DIR/clean_train_sev0.npz" "$TRAIN_EXPECT" "clean sev0"
for c in "${CORRUPTIONS[@]}"; do
  for s in "${SEVERITIES[@]}"; do
    check "$CACHE_DIR/${c}_train_sev${s}.npz" "$TRAIN_EXPECT" "$c sev$s"
  done
done
echo ""
echo "VAL split (expect $VAL_EXPECT records each)"
echo "------------------------------------------------------"
check "$CACHE_DIR/clean_val_sev0.npz" "$VAL_EXPECT" "clean sev0"
for c in "${CORRUPTIONS[@]}"; do
  for s in "${SEVERITIES[@]}"; do
    check "$CACHE_DIR/${c}_val_sev${s}.npz" "$VAL_EXPECT" "$c sev$s"
  done
done
echo ""
echo "======================================================"
total=$((pass + fail + missing))
echo "  OK:      $pass / $total"
echo "  BAD:     $fail"
echo "  MISSING: $missing"
echo "======================================================"

if [ "$fail" -eq 0 ] && [ "$missing" -eq 0 ]; then
  echo ""
  echo "COMPLETE -- all 20 caches present with correct record counts."
  echo "Stage B (LCRE training) is unblocked."
  echo ""
  echo "  train: 10 conditions x $TRAIN_EXPECT = $((10 * TRAIN_EXPECT)) samples"
  echo "  val:   10 conditions x $VAL_EXPECT = $((10 * VAL_EXPECT)) samples (benchmark only)"
  exit 0
else
  echo ""
  echo "INCOMPLETE -- resolve the above before starting Stage B."
  exit 1
fi