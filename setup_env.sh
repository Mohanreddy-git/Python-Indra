#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install pandas numpy openpyxl

# Run the conversion script
python convert_embeddings.py