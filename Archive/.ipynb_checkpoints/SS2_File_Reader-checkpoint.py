import os

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

def load_ss2_folder(folder_path):
    X = []
    Y = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".ss2"):
            filepath = os.path.join(folder_path, filename)
            aa_seq, ss_seq = parse_ss2_file(filepath)
            X.append(aa_seq)
            Y.append(ss_seq)
    return X, Y

# Example usage
folder = "/Users/osim/Downloads/Protein Structures"
X, Y = load_ss2_folder(folder)

print("Number of proteins:", len(X))
print("First AA sequence:", X[2])
print("First SS sequence:", Y[2])
