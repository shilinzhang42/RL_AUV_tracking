#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for AUV diffusion training.
# Usage examples:
#   ./train_auv_diffusion.sh fm
#   ./train_auv_diffusion.sh ddpm
#   ./train_auv_diffusion.sh fm --device cuda:1
#   ./train_auv_diffusion.sh ddpm --overrides "training.num_epochs=30 logging.mode=offline"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE=""
DEVICE=""
OVERRIDES=""
DRY_RUN=0

print_help() {
    cat <<'EOF'
Usage:
  ./train_auv_diffusion.sh <fm|ddpm> [options]

Options:
  --device <device>      Override training device, e.g. cuda:0 / cuda:1 / cpu
  --overrides "k=v ..."  Extra Hydra overrides, space separated and quoted
  --dry-run              Print command only, do not execute
  -h, --help             Show this help message

Examples:
  ./train_auv_diffusion.sh fm
  ./train_auv_diffusion.sh ddpm --device cuda:1
  ./train_auv_diffusion.sh fm --overrides "training.num_epochs=20 logging.mode=offline"
EOF
}

if [[ $# -lt 1 ]]; then
    print_help
    exit 1
fi

MODE="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="${2:-}"
            if [[ -z "$DEVICE" ]]; then
                echo "[ERROR] --device requires a value."
                exit 1
            fi
            shift 2
            ;;
        --overrides)
            OVERRIDES="${2:-}"
            if [[ -z "$OVERRIDES" ]]; then
                echo "[ERROR] --overrides requires a value."
                exit 1
            fi
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

case "$MODE" in
    fm)
        TRAIN_SCRIPT="diffusion_policy/workspace/train_flow_matching_unet_image_workspace_custom.py"
        ;;
    ddpm)
        TRAIN_SCRIPT="diffusion_policy/workspace/train_ddpm_unet_image_workspace_auv.py"
        ;;
    *)
        echo "[ERROR] MODE must be 'fm' or 'ddpm', got: $MODE"
        print_help
        exit 1
        ;;
esac

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

CMD=(python "$TRAIN_SCRIPT")

if [[ -n "$DEVICE" ]]; then
    CMD+=("training.device=$DEVICE")
fi

if [[ -n "$OVERRIDES" ]]; then
    # shellcheck disable=SC2206
    EXTRA_OVERRIDES=($OVERRIDES)
    CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "================================================="
echo "AUV diffusion training launcher"
echo "Mode:   $MODE"
echo "Script: $TRAIN_SCRIPT"
if [[ -n "$DEVICE" ]]; then
    echo "Device override: $DEVICE"
fi
if [[ -n "$OVERRIDES" ]]; then
    echo "Extra overrides: $OVERRIDES"
fi
echo "Command: ${CMD[*]}"
echo "================================================="

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] Command not executed."
    exit 0
fi

"${CMD[@]}"
