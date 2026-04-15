#!/bin/bash
# Activation script for Lyapunov Neural Control environment

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lnc

# Set up Python path for verification
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"

echo "✓ Conda environment 'lnc' activated"
echo "✓ PYTHONPATH configured for verification"
echo ""
echo "You can now run:"
echo "  - Training: python examples/pendulum_state_training.py"
echo "  - Verification: cd verification && python abcrown.py --config <config_file>"


