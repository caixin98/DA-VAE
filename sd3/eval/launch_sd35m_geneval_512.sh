#!/usr/bin/env bash
set -euo pipefail

# Launch the GenEval pipeline on one GPU via the cluster's `gpu1` alias.
# This avoids nested srun and keeps the worker script (`run_sd35m_geneval_512.sh`) pure.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defensive: some cluster scripts assume these exist under `set -u`.
export PS1="${PS1-}"
export COLORTERM="${COLORTERM-}"
export XTERM_VERSION="${XTERM_VERSION-}"

# Load gpuX aliases (srun wrappers).
# On this cluster, sourced scripts may reference unset variables; temporarily disable nounset.
set +u
# source your .bashrc if needed
set -u

# `audit.sh` on this cluster references ZSH_VERSION under `set -u`.
# Exporting ZSH_VERSION prevents that crash; we don't source conda.sh in the job, so it's safe.
LOGDIR="${REPO_ROOT}/eval/outputs/geneval/_slurm_logs"
mkdir -p "${LOGDIR}"

# Default to gpu2 to avoid contention when gpu1 is already occupied.
GPU_RUNNER="${GPU_RUNNER:-gpu2}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"

${GPU_RUNNER} \
  --export=ALL,ZSH_VERSION=1,PS1= \
  --time="${SLURM_TIME}" \
  --output="${LOGDIR}/sd35m_geneval_%j.out" \
  --error="${LOGDIR}/sd35m_geneval_%j.err" \
  env -u SHELLOPTS bash "${REPO_ROOT}/eval/run_sd35m_geneval_512.sh"


