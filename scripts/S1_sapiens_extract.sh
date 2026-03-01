#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
POSE_SCRIPT="$SCRIPT_DIR/../sapiens/lite/scripts/demo/torchscript/pose_keypoints133.sh"
bash "$POSE_SCRIPT" --input "${ROOT_PATH}" --output "${OUTPUT_PATH}"