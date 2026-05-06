#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/proto-tools:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  examples/scripts/slurm/launch_epigenomics_morse_proto_live.sh \
    --left-flank path/to/left.fa \
    --right-flank path/to/right.fa \
    --output-dir path/to/run_dir \
    [additional epigenomics_morse_proto.py args...]

This wrapper is intentionally thin: pass runtime configuration as command-line
arguments to examples/scripts/epigenomics_morse_proto.py.
EOF
  exit 1
fi

OUTPUT_DIR=""
ARGS=("$@")
for ((idx = 0; idx < ${#ARGS[@]}; idx++)); do
  if [[ "${ARGS[idx]}" == "--output-dir" ]]; then
    if (( idx + 1 >= ${#ARGS[@]} )); then
      echo "Missing value for --output-dir" >&2
      exit 2
    fi
    OUTPUT_DIR="${ARGS[idx + 1]}"
    break
  fi
done

if [[ -n "${OUTPUT_DIR}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  python -u examples/scripts/epigenomics_morse_proto.py "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/launch.log"
else
  exec python -u examples/scripts/epigenomics_morse_proto.py "${ARGS[@]}"
fi
