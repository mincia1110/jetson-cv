#!/usr/bin/env bash
# Build on the target Jetson; no application-time conversion.
set -euo pipefail
[[ $# -ge 2 ]] || { echo 'Usage: bash scripts/build_engine.sh MODEL.onnx NEW_OUTPUT_DIR [trtexec build args...]' >&2; exit 2; }
model=$1
out=$2
shift 2
[[ -f "$model" ]] || { echo "Missing model: $model" >&2; exit 1; }
trt_exec=$(command -v trtexec || true)
trt_exec=${trt_exec:-/usr/src/tensorrt/bin/trtexec}
[[ -x "$trt_exec" ]] || { echo 'trtexec missing. Install JetPack development components.' >&2; exit 1; }
# Keep model/output selection controlled here; extra args configure the builder.
for arg in "$@"; do
  case "$arg" in
    --onnx*|--saveEngine*|--loadEngine*|--useDLACore*|--allowGPUFallback*|--skipInference*)
      echo "Reserved option: $arg" >&2; exit 2 ;;
  esac
done
mkdir -p "$(dirname "$out")"
mkdir "$out" # Fail instead of overwriting an earlier engine or failed-build evidence.
printf '%q ' "$trt_exec" --onnx="$model" --saveEngine="$out/model.engine" --skipInference --noTF32 "$@" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
sha256sum "$model" > "$out/onnx.sha256"
"$trt_exec" --onnx="$model" --saveEngine="$out/model.engine" --skipInference --noTF32 "$@" 2>&1 | tee "$out/build.log"
[[ -s "$out/model.engine" ]] || { echo 'Build did not produce an engine.' >&2; exit 1; }
sha256sum "$out/model.engine" > "$out/engine.sha256"
echo 'Engine built. Validate with trt_probe.py before connecting the application.'
