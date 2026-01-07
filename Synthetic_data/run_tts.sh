#!/bin/bash
# Wrapper script to run syn_para.py with cleaned LD_LIBRARY_PATH
# This removes problematic Singularity library paths that cause GLIBC errors

# Prepend system library paths to take precedence over Singularity libs
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

pip install --force-reinstall --no-deps antlr4-python3-runtime==4.9.3
pip install --force-reinstall --no-deps transformers==4.52.1

echo "Cleaned LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Unset SSL_CERT_FILE to avoid SSL certificate path issues
unset SSL_CERT_FILE

echo "Running syn_para.py..."
echo "================================"

# Run the Python script with all arguments passed to this script
python syn_para.py "$@"
