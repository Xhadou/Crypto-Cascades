#!/bin/bash
set -e

echo "=== Crypto Cascades Environment Setup ==="

# Create virtual environment
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$WINDIR" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify critical packages
python -c "import networkx; print(f'NetworkX: {networkx.__version__}')"
python -c "import pyarrow; print(f'PyArrow: {pyarrow.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import yaml; print(f'PyYAML installed successfully')"

echo "=== Setup Complete ==="
echo "Activate with: source venv/Scripts/activate (Windows) or source venv/bin/activate (Unix)"
