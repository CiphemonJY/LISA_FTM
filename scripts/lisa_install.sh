#!/bin/bash
# LISA Install Script
# Usage: bash scripts/lisa_install.sh

echo "Installing LISA Federated Learning..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    sudo apt update && sudo apt install -y python3 python3-pip
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install torch transformers peft huggingface_hub accelerate

# Verify
python3 -c "import torch; import transformers; import peft" 2>/dev/null
if [ $? -eq 0 ]; then
    echo ""
    echo "Installation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Train locally: python3 train_7b_simple.py --steps 100"
    echo "  2. For detailed setup: See docs/JETSON_SETUP.md"
else
    echo "Installation may have issues. Check errors above."
fi
