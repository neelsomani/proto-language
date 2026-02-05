#!/bin/bash
#SBATCH --job-name=complex_diversification
#SBATCH --output=logs/slurm/complex_div_%x_%A_%a.log
#SBATCH --error=logs/slurm/complex_div_%x_%A_%a.log
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu,gpu_high_mem,preemptible
#SBATCH --requeue

# =============================================================================
# SLURM Job Template for Complex Diversification
# =============================================================================
#
# This template is designed to be run from the proto-language root directory.
#
# Usage (single job):
#   sbatch --export=PROGRAM_SCRIPT=path/to/program.py slurm/job_template.sbatch
#
# Usage (array job - handled by submit_all.sh):
#   sbatch --array=0-N%20 slurm/job_template.sbatch
#
# Environment variables:
#   COMPLEX_PROGRAMS_DIR: Path to the complex programs directory
#                         (default: examples/scripts/human_systems_claude)
#
# =============================================================================

set -e  # Exit on error

# Default paths (can be overridden via environment)
COMPLEX_PROGRAMS_DIR="${COMPLEX_PROGRAMS_DIR:-examples/scripts/human_systems_claude}"

# Print job info
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Complex programs dir: $COMPLEX_PROGRAMS_DIR"
echo "=============================================="

# Activate conda environment (adjust as needed)
# source ~/.bashrc
# conda activate proto_language

# Validate complex programs directory exists
if [ ! -d "$COMPLEX_PROGRAMS_DIR" ]; then
    echo "ERROR: Complex programs directory not found: $COMPLEX_PROGRAMS_DIR"
    echo "Make sure you're running from the proto-language root directory"
    exit 1
fi

# If running as array job, get program script and filename from manifest
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    MANIFEST="$COMPLEX_PROGRAMS_DIR/manifest.json"
    if [ ! -f "$MANIFEST" ]; then
        echo "ERROR: Manifest not found at $MANIFEST"
        exit 1
    fi

    # Read program path and filename_base from manifest using task ID
    read PROGRAM_SCRIPT FILENAME_BASE < <(python3 -c "
import json
with open('$MANIFEST') as f:
    manifest = json.load(f)
program = manifest['programs'][$SLURM_ARRAY_TASK_ID]
script = program['program_path'].replace('./', '$COMPLEX_PROGRAMS_DIR/')
print(script, program['filename_base'])
")
fi

# Validate program script
if [ -z "$PROGRAM_SCRIPT" ]; then
    echo "ERROR: PROGRAM_SCRIPT not set"
    exit 1
fi

if [ ! -f "$PROGRAM_SCRIPT" ]; then
    echo "ERROR: Program script not found: $PROGRAM_SCRIPT"
    exit 1
fi

# Create timestamped run directory and export for Python
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S_%6N)
export RUN_OUTPUT_DIR="$COMPLEX_PROGRAMS_DIR/outputs/$FILENAME_BASE/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_OUTPUT_DIR"
PROGRAM_LOG="$RUN_OUTPUT_DIR/program.log"

echo "Running program: $PROGRAM_SCRIPT"
echo "Run output dir: $RUN_OUTPUT_DIR"
echo "Program log: $PROGRAM_LOG"
echo "=============================================="

# Run the program, redirecting its output to the run directory
python3 "$PROGRAM_SCRIPT" 2>&1 | tee "$PROGRAM_LOG"

exit_code=${PIPESTATUS[0]}

echo "=============================================="
echo "Program finished with exit code: $exit_code"
echo "Date: $(date)"
echo "=============================================="

exit $exit_code
