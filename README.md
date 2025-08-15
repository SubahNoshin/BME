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
2. Place corresponding **SS2 secondary structure files** in a separate folder (e.g., `protein structures`), with filenames matching sequence IDs in the FASTA file.

## Data Preprocessing

The preprocessing pipeline converts raw protein sequences and secondary structure annotations into a format suitable for CNN training:

1. **Amino Acid Encoding**
   - Encode each amino acid (20 common residues: `ACDEFGHIKLMNPQRSTVWY`) as a **one-hot vector**.  
   - Each protein sequence becomes a **matrix of shape `(max_len, 20)`**.  
   - Sequences shorter than `max_len` are **padded with zeros**.

2. **Secondary Structure Mapping**
   - Map secondary structure labels to integers:
     - `H` → 0 (Helix)  
     - `C` → 1 (Coil)  
     - `B` → 2 (Strand/Beta-sheet)  
   - Shorter sequences are **padded with `1` (Coil)** to match `max_len`.

3. **One-hot Encoding of Labels**
   - Use **Keras `to_categorical`** to convert integer labels to one-hot format.  
   - Resulting shape: `(num_sequences, max_len, 3)`.

4. **Train/Validation/Test Split**
   - Split dataset into:
     - **Training:** 70%  
     - **Validation:** 15%  
     - **Test:** 15%  


