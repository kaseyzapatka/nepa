#!/bin/bash

# Create conda environment with Python 3.12
echo "Creating conda environment 'textanalysis' with Python 3.12..."
conda create -n textanalysis python=3.12 -y

# Activate the environment
echo "Activating environment..."
conda activate textanalysis

# Install basic data science libraries via conda (faster and more reliable)
echo "Installing basic Python libraries..."
conda install -y pandas numpy matplotlib seaborn scikit-learn scipy jupyter ipykernel pyarrow datasets

# Install text analysis packages via pip
echo "Installing text analysis libraries..."
pip install scattertext wordcloud spacy

# Download spacy language model (English)
echo "Downloading spacy English language model..."
python -m spacy download en_core_web_sm

# Register the kernel for Jupyter
echo "Registering kernel for Jupyter..."
python -m ipykernel install --user --name textanalysis --display-name "Python 3.12 (textanalysis)"

echo ""
echo "✓ Environment 'textanalysis' created successfully!"
echo ""
echo "To use this environment:"
echo "  1. Run: conda activate textanalysis"
echo "  2. In Positron/Jupyter, select kernel: 'Python 3.12 (textanalysis)'"
echo ""
echo "Installed packages:"
conda list | grep -E "pandas|numpy|matplotlib|spacy|scattertext|wordcloud"