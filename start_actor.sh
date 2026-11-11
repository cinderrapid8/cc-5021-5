#!/usr/bin/env bash
# 虚拟舞台演员启动脚本：无编辑器环境下运行发布包并执行真实自检。
# 自检失败时退出非零，且不会保留证据文件；调用方不得采用本次产物。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
OUT="${ACTOR_OUT:-${ROOT}/out/actor_runtime}"

exec "${PY}" "${ROOT}/run_actor.py" \
    --package "${ROOT}/release" \
    --out "${OUT}" \
    "$@"
