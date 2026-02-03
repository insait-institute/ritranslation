#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --model_name NAME        Model name or path (required unless MODEL_NAME env set)
  --model NAME             Alias for --model_name
  --host HOST              Host to bind (default: ${HOST:-0.0.0.0})
  --port PORT              Port (default: ${PORT:-8080})
  --api_key KEY            API key (default: ${API_KEY:-token-abc123})
  --dtype DTYPE            Data type (default: ${DTYPE:-auto})
  --tp_size SIZE           Tensor parallel size (default: auto-calculated based on GPU count)
  --log_level LEVEL        Log level (default: ${LOG_LEVEL:-info})
  -h, --help               Show this help
EOF
}

source test-env/bin/activate

# Defaults (can be overridden by env vars or CLI flags)
MODEL_NAME="${MODEL_NAME:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-token-abc123}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Parse long options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)
            MODEL_NAME="$2"; shift 2;;
        --model)
            MODEL_NAME="$2"; shift 2;;
        --host)
            HOST="$2"; shift 2;;
        --port)
            PORT="$2"; shift 2;;
        --api_key)
            API_KEY="$2"; shift 2;;
        --dtype)
            DTYPE="$2"; shift 2;;
        --tp_size)
            TP_SIZE="$2"; shift 2;;
        --log_level)
            LOG_LEVEL="$2"; shift 2;;
        -h|--help)
            usage; exit 0;;
        *)
            echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model_name is required (or set MODEL_NAME env var)." >&2
    usage
    exit 2
fi

# Detect number of available GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# If TP_SIZE not specified, find the closest valid tensor parallel size (must divide 64 for most models)
# Common divisors of 64: 1, 2, 4, 8, 16, 32, 64
if [[ -z "$TP_SIZE" ]]; then
    find_closest_divisor() {
        local count=$1
        local divisors=(1 2 4 8 16 32 64)
        local closest=1
        local min_diff=999
        
        for div in "${divisors[@]}"; do
            if [[ $div -le $count ]]; then
                diff=$((count - div))
                if [[ $diff -lt $min_diff ]]; then
                    min_diff=$diff
                    closest=$div
                fi
            fi
        done
        echo $closest
    }
    
    TP_SIZE=$(find_closest_divisor $GPU_COUNT)
    echo "Detected $GPU_COUNT GPU(s). Auto-calculated tensor parallel size: $TP_SIZE"
else
    echo "Using user-specified tensor parallel size: $TP_SIZE"
fi

exec vllm serve \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP_SIZE" 