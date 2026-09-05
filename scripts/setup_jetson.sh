#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || { echo 'Run on Jetson ARM64 Linux.' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || { echo 'Expected Ubuntu 22.04.' >&2; exit 1; }
l4t=$(dpkg-query -W -f='${Version}' nvidia-l4t-core)
[[ "$l4t" == 36.4.4-* ]] || { echo "Expected L4T 36.4.4, got $l4t. Review baseline first." >&2; exit 1; }
sudo apt-get update
sudo apt-get install -y nvidia-jetpack python3-venv python3-pip python3-opencv python3-numpy v4l-utils usbutils
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'onnx==1.16.2' 'polygraphy==0.49.9'
.venv/bin/python -m pip check
mkdir -p artifacts
echo 'Base installed. Activate with: source .venv/bin/activate'
