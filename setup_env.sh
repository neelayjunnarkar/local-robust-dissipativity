#!/bin/bash
# Activation script for Lyapunov Neural Control environment

set -e

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment unless one is already active.
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
	active_env="$CONDA_DEFAULT_ENV"
elif [[ -n "${LNC_ENV_NAME:-}" ]]; then
	conda activate "$LNC_ENV_NAME"
	active_env="$LNC_ENV_NAME"
else
	conda activate lnc
	active_env="lnc"
fi

# Set up Python path for verification.
export PYTHONPATH="${repo_root}:${repo_root}/alpha-beta-CROWN:${repo_root}/alpha-beta-CROWN/auto_LiRPA:${repo_root}/alpha-beta-CROWN/complete_verifier${PYTHONPATH:+:${PYTHONPATH}}"

echo "✓ Conda environment '$active_env' ready"
echo "✓ PYTHONPATH configured for verification"
echo ""
echo "You can now run:"
echo "  - Training: python examples/pendulum_state_training.py"
echo "  - Verification: cd verification && python abcrown.py --config <config_file>"


