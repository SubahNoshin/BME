# Protein Secondary Structure Prediction with CNN

This project implements a **1D Convolutional Neural Network (CNN)** to predict **protein secondary structures** from amino acid sequences. The pipeline includes data preprocessing, model training, evaluation, and visualization of training dynamics.

---

## Project Structure

---

## Dependencies

- Python 3.9+
- TensorFlow / Keras
- NumPy
- SciPy
- Biopython
- scikit-learn
- matplotlib
- seaborn
- imageio

Install dependencies via pip:

```bash
pip install tensorflow numpy biopython scikit-learn matplotlib seaborn imageio

# Protein Secondary Structure Prediction with CNN

This project uses a **1D Convolutional Neural Network (CNN)** to predict protein secondary structures from amino acid sequences. It includes preprocessing, model training, evaluation, and visualization of Conv1D weight evolution.

---

## Data Preparation

1. Place your **FASTA sequences** in the `data/` folder (e.g., `sequence.fasta`).  
2. Place corresponding **SS2 secondary structure files** in a separate folder (e.g., `data/ss2/`), with filenames matching sequence IDs in the FASTA file.

---

## Usage

### 1. Preprocessing & One-hot Encoding
```bash
python scripts/preprocess.py

