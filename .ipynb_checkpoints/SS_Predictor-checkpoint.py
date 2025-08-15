import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import matplotlib.pyplot as plt
import os

# -------------------------------
# Step 1: Load and encode FASTA
# -------------------------------

def one_hot_encode_seq(seq, max_len):
    """One-hot encode protein sequence (Amino acids A-Z)"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 common
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    encoding = np.zeros((max_len, len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if i >= max_len:
            break
        if aa in aa_to_int:
            encoding[i, aa_to_int[aa]] = 1.0
    return encoding

def parse_ss2_file(filepath):
    aa_seq = []
    ss_seq = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                aa_seq.append(parts[1])  # amino acid
                ss_seq.append(parts[2])  # secondary structure
    ss_seq = [s.replace('E', 'B') for s in ss_seq]
    return "".join(aa_seq), "".join(ss_seq)

def get_secondary_structure_from_ss2_file(folder_path, filename):
    
    filepath = os.path.join(folder_path, filename)
    _, ss_seq = parse_ss2_file(filepath)
    return ss_seq

# Load sequences from FASTA
fasta_path = "sequence.fasta"  # your file
ss_folder_path = "/Users/osim/Downloads/Protein Structures"
records = list(SeqIO.parse(fasta_path, "fasta"))
max_len = max(len(record.seq) for record in records)  # pad to longest sequence

X = np.array([one_hot_encode_seq(str(record.seq), max_len) for record in records])
y_ss = np.array([get_secondary_structure_from_ss2_file(ss_folder_path, record.id + ".ss2") for record in records])

# Map secondary structure to integer labels
ss_map = {'H': 0, 'C': 1, 'B': 2}
y_labels = np.array([
    [ss_map.get(s, 1) for s in ss][:max_len] + [1]*(max_len - len(ss))  # pad with 'C' (1) if needed
    for ss in y_ss
])

num_classes = 3
y_cat = to_categorical(y_labels, num_classes=num_classes)

# -------------------------------
# Step 3: Train/Val/Test split
# -------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# -------------------------------
# Step 4: Build CNN Model
# -------------------------------
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=(max_len, X.shape[2])),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation="relu", padding="same"),
    Dropout(0.3),
    Conv1D(num_classes, kernel_size=1, activation="softmax", padding="same")  # per-residue prediction
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# -------------------------------
# Step 5: Train
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8
)

# -------------------------------
# Step 6: Evaluate
# -------------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# -------------------------------
# Step 7: Plot loss and accuracy
# -------------------------------
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss over Epochs")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy over Epochs")
plt.show()