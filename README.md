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
### Model Training

The CNN model is trained using the following steps:

1. **Build the 1D CNN Model**  
   - Architecture: `Conv1D → Dropout → Conv1D → Dropout → Conv1D (softmax per residue)`.  
   - Designed for **per-residue secondary structure prediction**.

2. **Compile the Model**  
   - **Loss function:** `categorical_crossentropy`  
   - **Optimizer:** `adam`  
   - **Metric:** `accuracy`

3. **Train the Model**  
   - Use the training dataset with a validation set for monitoring performance.  
   - Batch size and number of epochs are configurable.

4. **Generate Accuracy and Loss Plots**  
   - Training and validation accuracy/loss are plotted over epochs to visualize model learning.

5. **Visualize Conv1D Kernel Weights**  
   - At the end of each epoch, the first Conv1D layer weights are visualized.  
   - A **GIF of weight evolution** is saved: `weights_frames/weights_evolution.gif`.

---

### Model Evaluation

After training, the model is evaluated on the test dataset as follows:

1. **Load the Trained Model**  
   - Load saved model weights and architecture for predictions.

2. **Prepare Test Data**  
   - Test sequences are one-hot encoded and padded to match the maximum sequence length.

3. **Predict Secondary Structures**  
   - The model outputs a probability distribution per residue over the classes (`H`, `C`, `B`).  
   - The predicted class for each residue is the one with the highest probability.

4. **Flatten Sequences for Analysis**  
   - Flatten predicted and true labels to a 1D array.  
   - Remove padding to focus only on real residues.

5. **Compute Per-Residue Accuracy**  
   - Measures the proportion of correctly predicted residues.

6. **Generate Confusion Matrix**  
   - Shows how residues of each true class are predicted, highlighting misclassifications.

7. **Produce Classification Report**  
   - Reports **precision**, **recall**, and **F1-score** for each class (`H`, `C`, `B`).

8. **Visualize Confusion Matrix**  
   - Displayed as a heatmap for easy visual analysis of predictions versus true labels.
![image alt](https://github.com/SubahNoshin/BME/blob/main/accuracy_curve.png)
