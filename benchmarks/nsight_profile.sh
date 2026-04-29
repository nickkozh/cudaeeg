#!/usr/bin/env bash
# NSight Systems wrapper. Captures kernel timeline, CUDA API trace, GPU metrics.
# Output goes to benchmarks/nsight_out/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-${ROOT}/benchmarks/nsight_out}"
CHANNELS="${CHANNELS:-32}"
mkdir -p "$OUT"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found. On JetPack 6: sudo apt install nsight-systems-cli"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
REP="$OUT/cuda_eeg_prep_${CHANNELS}ch_${STAMP}"

echo "Profiling cuda-eeg-prep at ${CHANNELS} channels…"
nsys profile \
  --output "$REP" \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  --force-overwrite=true \
  -- python "$ROOT/benchmarks/bench.py" --channels "$CHANNELS"

echo
echo "Generating kernel summary CSV…"
nsys stats --report cuda_kern_exec_trace --format csv \
  "${REP}.nsys-rep" > "${REP}_kernel_summary.csv" || true

echo
echo "Done."
echo "  Timeline:        ${REP}.nsys-rep"
echo "  Kernel summary:  ${REP}_kernel_summary.csv"
echo
echo "Open the .nsys-rep in NSight Systems GUI to inspect the timeline."
