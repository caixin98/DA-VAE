#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${PYTHON:=python}"

DEFAULT_INPUT_DIR="${PROJECT_ROOT}/outputs/sd35_1024_seedvr2_sr_by_cat"
DEFAULT_REFERENCE_DIR="${PROJECT_ROOT}/datasets/mjhq30k"
# If empty, we compute FID by directly comparing against --reference (your dataset folder).
DEFAULT_STATS_NAME=""
DEFAULT_DATASET_RES="1024"
DEFAULT_JSON_NAME="fid.json"
DEFAULT_CACHE_DIR=""
DEFAULT_FLATTEN="0"
DEFAULT_MAX_IMAGES="0"
DEFAULT_SEED="0"
DEFAULT_CREATE_STATS="0"
DEFAULT_STATS_MODE="clean"

INPUT_DIR="${INPUT_DIR:-$DEFAULT_INPUT_DIR}"
REFERENCE_DIR="${REFERENCE_DIR:-$DEFAULT_REFERENCE_DIR}"
STATS_NAME="${STATS_NAME:-$DEFAULT_STATS_NAME}"
DATASET_RES="${DATASET_RES:-$DEFAULT_DATASET_RES}"
JSON_NAME="${JSON_NAME:-$DEFAULT_JSON_NAME}"
CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE_DIR}"
FLATTEN="${FLATTEN:-$DEFAULT_FLATTEN}"
MAX_IMAGES="${MAX_IMAGES:-$DEFAULT_MAX_IMAGES}"
SEED="${SEED:-$DEFAULT_SEED}"
CREATE_STATS="${CREATE_STATS:-$DEFAULT_CREATE_STATS}"
STATS_MODE="${STATS_MODE:-$DEFAULT_STATS_MODE}"

usage() {
	cat <<EOF
Usage:

  ${0##*/} --input PATH [--reference PATH] [--stats-name NAME] [--dataset-res N] [--json-name NAME] [--flatten|--no-flatten] [--cache-dir PATH]

Compute FID for a generated images directory and save the result JSON in that directory.

Options:
  --input PATH         Directory containing generated images (required)
  --reference PATH     Reference image directory (default: ${REFERENCE_DIR})
  --stats-name NAME    CleanFID stats name to reuse (default: ${STATS_NAME}; empty => compare to --reference)
  --dataset-res N      Dataset resolution tag when using --stats-name (default: ${DATASET_RES})
  --json-name NAME     Output JSON filename inside input dir (default: ${JSON_NAME})
  --cache-dir PATH     Cache dir for flattened symlink trees (default: <input>/__fid_cache__)
  --flatten            Enable flattening into a cached, single-level symlink tree (off by default)
  --no-flatten         Disable flattening (default); pass directories directly to CleanFID
  --max-images N       If >0, sample at most N images (applied when flattening; default: ${MAX_IMAGES})
  --seed N             RNG seed used when sampling images (default: ${SEED})
  --create-stats       When using --stats-name, create stats from --reference before computing FID (default: ${CREATE_STATS})
  --stats-mode MODE    CleanFID stats mode when using --stats-name (default: ${STATS_MODE})
  -h, --help           Show this message and exit

Notes:
- Uses eval/compute_fid.py under the hood.
- For "by_cat" folder structures, consider passing --flatten to build/reuse a cached symlink tree (can speed up repeated runs).
- Result JSON will be written to: <input_dir>/<json-name>
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--input)
			INPUT_DIR="$2"
			shift 2
			;;
		--reference)
			REFERENCE_DIR="$2"
			shift 2
			;;
		--stats-name)
			STATS_NAME="$2"
			shift 2
			;;
		--dataset-res)
			DATASET_RES="$2"
			shift 2
			;;
		--json-name)
			JSON_NAME="$2"
			shift 2
			;;
		--cache-dir)
			CACHE_DIR="$2"
			shift 2
			;;
		--flatten)
			FLATTEN="1"
			shift 1
			;;
		--no-flatten)
			FLATTEN="0"
			shift 1
			;;
		--max-images)
			MAX_IMAGES="$2"
			shift 2
			;;
		--seed)
			SEED="$2"
			shift 2
			;;
		--create-stats)
			CREATE_STATS="1"
			shift 1
			;;
		--stats-mode)
			STATS_MODE="$2"
			shift 2
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "[FID-ONLY] Unknown argument: $1" >&2
			usage
			exit 2
			;;
	esac
done

if [[ -z "${INPUT_DIR}" ]]; then
	echo "[FID-ONLY] Missing required --input PATH" >&2
	usage
	exit 2
fi

# Helper: flatten a directory tree into a single directory of images using symlinks (fallback: copy).
flatten_images() {
	local src="$1"
	local dst="$2"
	local max_images="${3:-0}"
	local seed="${4:-0}"
	"${PYTHON}" - "${src}" "${dst}" "${max_images}" "${seed}" <<'PY'
import hashlib
import random
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
max_images = int(sys.argv[3])
seed = int(sys.argv[4])
dst.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
exclude_parts = {"__staging__", "_slurm_logs", "__fid_cache__", "__corrupted__"}

paths = []
for p in src.rglob("*"):
    if not p.is_file():
        continue
    if p.suffix.lower() not in exts:
        continue
    rel = p.relative_to(src)
    if any(part in exclude_parts for part in rel.parts):
        continue
    paths.append(p)

paths.sort(key=lambda p: str(p))
if max_images > 0 and len(paths) > max_images:
    rng = random.Random(seed)
    paths = rng.sample(paths, max_images)
    paths.sort(key=lambda p: str(p))

count = 0
for p in paths:
    h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:16]
    out = dst / f"{h}{p.suffix.lower()}"
    if out.exists():
        continue
    try:
        out.symlink_to(p)
    except Exception:
        shutil.copy2(p, out)
    count += 1

print(count)
PY
}

count_flat_images() {
	local d="$1"
	"${PYTHON}" - "${d}" <<'PY'
import sys
from pathlib import Path

d = Path(sys.argv[1]).expanduser().resolve()
if not d.exists() or not d.is_dir():
    print(0)
    raise SystemExit(0)

exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
cnt = 0
for p in d.iterdir():
    if p.is_file() and p.suffix.lower() in exts:
        cnt += 1
print(cnt)
PY
}

# Resolve absolute paths
INPUT_PATH=$("${PYTHON}" - "${INPUT_DIR}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)

REFERENCE_PATH=$("${PYTHON}" - "${REFERENCE_DIR}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)

if [[ ! -d "${INPUT_PATH}" ]]; then
	echo "[FID-ONLY] Input directory not found: ${INPUT_PATH}" >&2
	exit 1
fi

if [[ -z "${STATS_NAME}" ]]; then
	if [[ -z "${REFERENCE_PATH}" || ! -d "${REFERENCE_PATH}" ]]; then
		echo "[FID-ONLY] Reference directory not found: ${REFERENCE_DIR}" >&2
		exit 1
	fi
fi

echo "[FID-ONLY] Input: ${INPUT_PATH}" >&2
if [[ -n "${STATS_NAME}" ]]; then
	echo "[FID-ONLY] Stats name: ${STATS_NAME} (dataset_res=${DATASET_RES})" >&2
else
	echo "[FID-ONLY] Reference: ${REFERENCE_PATH}" >&2
fi

INPUT_FOR_FID="${INPUT_PATH}"
REFERENCE_FOR_FID="${REFERENCE_PATH}"

if [[ "${FLATTEN}" == "1" ]]; then
	if [[ -z "${CACHE_DIR}" ]]; then
		CACHE_DIR="${INPUT_PATH}/__fid_cache__"
	fi

	CACHE_SUFFIX="full"
	if [[ "${MAX_IMAGES}" != "0" ]]; then
		CACHE_SUFFIX="n${MAX_IMAGES}_s${SEED}"
	fi

	GEN_FLAT="${CACHE_DIR}/generated_flat_${CACHE_SUFFIX}"
	REF_FLAT="${CACHE_DIR}/reference_flat_${CACHE_SUFFIX}"

	echo "[FID-ONLY] Flatten enabled. Cache dir: ${CACHE_DIR}" >&2
	# If sampling, ensure the cache dir contains exactly the sampled subset.
	if [[ "${MAX_IMAGES}" != "0" ]]; then
		rm -rf "${GEN_FLAT}" "${REF_FLAT}"
	fi
	mkdir -p "${GEN_FLAT}" "${REF_FLAT}"

	# Reuse existing full-cache to avoid re-walking 30k files every run.
	if [[ "${MAX_IMAGES}" == "0" ]]; then
		GEN_EXISTING="$(count_flat_images "${GEN_FLAT}")"
		if [[ "${GEN_EXISTING}" -gt 0 ]]; then
			echo "[FID-ONLY] Reusing cached flattened generated images: ${GEN_EXISTING} -> ${GEN_FLAT}" >&2
		else
			GEN_COUNT="$(flatten_images "${INPUT_PATH}" "${GEN_FLAT}" "${MAX_IMAGES}" "${SEED}")"
			echo "[FID-ONLY] Flattened generated images: ${GEN_COUNT} -> ${GEN_FLAT}" >&2
		fi
	else
		GEN_COUNT="$(flatten_images "${INPUT_PATH}" "${GEN_FLAT}" "${MAX_IMAGES}" "${SEED}")"
		echo "[FID-ONLY] Flattened generated images: ${GEN_COUNT} -> ${GEN_FLAT}" >&2
	fi

	# If we are comparing folder-vs-folder (no stats), or we are creating stats from a reference
	# directory, we need a flattened reference image directory too.
	if [[ -z "${STATS_NAME}" || "${CREATE_STATS}" == "1" ]]; then
		if [[ "${MAX_IMAGES}" == "0" ]]; then
			REF_EXISTING="$(count_flat_images "${REF_FLAT}")"
			if [[ "${REF_EXISTING}" -gt 0 ]]; then
				echo "[FID-ONLY] Reusing cached flattened reference images: ${REF_EXISTING} -> ${REF_FLAT}" >&2
			else
				REF_COUNT="$(flatten_images "${REFERENCE_PATH}" "${REF_FLAT}" "${MAX_IMAGES}" "${SEED}")"
				echo "[FID-ONLY] Flattened reference images: ${REF_COUNT} -> ${REF_FLAT}" >&2
			fi
		else
			REF_COUNT="$(flatten_images "${REFERENCE_PATH}" "${REF_FLAT}" "${MAX_IMAGES}" "${SEED}")"
			echo "[FID-ONLY] Flattened reference images: ${REF_COUNT} -> ${REF_FLAT}" >&2
		fi
	fi

	INPUT_FOR_FID="${GEN_FLAT}"
	REFERENCE_FOR_FID="${REF_FLAT}"
fi

FID_ARGS=("${INPUT_FOR_FID}")
if [[ -n "${STATS_NAME}" ]]; then
	FID_ARGS+=("--stats-name" "${STATS_NAME}" "--dataset-res" "${DATASET_RES}" "--stats-mode" "${STATS_MODE}")
	if [[ "${CREATE_STATS}" == "1" ]]; then
		FID_ARGS+=("--reference" "${REFERENCE_FOR_FID}" "--create-stats")
	fi
else
	# IMPORTANT: eval/compute_fid.py prefers stats-name if non-empty.
	# Force empty stats-name so it uses --reference mode.
	FID_ARGS+=("--reference" "${REFERENCE_FOR_FID}" "--stats-name" "")
fi

FID_OUTPUT=$(NCCL_DEBUG=WARN WORLD_SIZE= LOCAL_RANK= RANK= "${PYTHON}" "${PROJECT_ROOT}/eval/compute_fid.py" "${FID_ARGS[@]}")

printf '%s\n' "${FID_OUTPUT}"

# Extract the last numeric-looking line as the FID score
FID_SCORE=$(printf '%s\n' "${FID_OUTPUT}" | grep -Eo '^[0-9]+(\.[0-9]+)?$' | tail -n 1 || true)

if [[ -n "${FID_SCORE}" ]]; then
	echo "[FID-ONLY] Score: ${FID_SCORE}" >&2
else
	echo "[FID-ONLY] WARNING: Could not extract FID score from output" >&2
fi

# Write JSON result into the input directory
TIMESTAMP_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
OUTPUT_JSON="${INPUT_PATH}/${JSON_NAME}"
TMP_JSON="${OUTPUT_JSON}.tmp"

{
	echo "{"
	if [[ -n "${FID_SCORE}" ]]; then
		echo "  \"fid\": ${FID_SCORE},"
	else
		echo "  \"fid\": null,"
	fi
	echo "  \"generated\": \"${INPUT_PATH}\","
	if [[ -n "${STATS_NAME}" ]]; then
		echo "  \"reference\": null,"
	else
		echo "  \"reference\": \"${REFERENCE_PATH}\","
	fi
	echo "  \"flatten\": ${FLATTEN},"
	echo "  \"max_images\": ${MAX_IMAGES},"
	echo "  \"seed\": ${SEED},"
	if [[ -n "${CACHE_DIR}" ]]; then
		echo "  \"cache_dir\": \"${CACHE_DIR}\","
	else
		echo "  \"cache_dir\": null,"
	fi
	echo "  \"stats_name\": \"${STATS_NAME}\","
	echo "  \"dataset_res\": \"${DATASET_RES}\","
	echo "  \"stats_mode\": \"${STATS_MODE}\","
	echo "  \"create_stats\": ${CREATE_STATS},"
	echo "  \"computed_at\": \"${TIMESTAMP_UTC}\","
	echo "  \"source\": \"run_fid_only.sh\""
	echo "}"
} > "${TMP_JSON}"

mv -f "${TMP_JSON}" "${OUTPUT_JSON}"
echo "[FID-ONLY] Wrote JSON: ${OUTPUT_JSON}" >&2

exit 0


