#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]] || { echo 'Usage: bash scripts/trt_smoke.sh MODEL.onnx OUTPUT_DIR [trtexec args...]' >&2; exit 2; }
model=$1
out=$2
shift 2
[[ -f "$model" ]] || { echo "Missing model: $model" >&2; exit 1; }
trt_exec=$(command -v trtexec || true)
trt_exec=${trt_exec:-/usr/src/tensorrt/bin/trtexec}
[[ -x "$trt_exec" ]] || { echo 'trtexec missing. Install JetPack development components.' >&2; exit 1; }
mkdir -p "$out"
"$trt_exec" --onnx="$model" --saveEngine="$out/model.engine" --warmUp=500 --duration=3 --dumpLayerInfo "$@" 2>&1 | tee "$out/trtexec.log"
echo 'trtexec completed; inspect log. This does not validate VisualAD accuracy.'
