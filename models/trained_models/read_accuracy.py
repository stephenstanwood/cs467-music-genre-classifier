import sys
from pathlib import Path


"""
Evaluate and compare saved models using the GTZAN test dataset.

Currently, this script only supports models trained on
Mel spectrogram data. It loads the saved genre labels,
extracts the test split, and evaluates each saved model
to compare accuracy and loss.
"""

# ================= PROJECT PATH ======================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

# ==================== LOAD DATA ======================
from models.functions import *


data = np.load(
    PROJECT_ROOT / "Data" / "gtzan_mels.npz",
    allow_pickle=True
)

X = data["X"].astype(np.float32)
labels = data["label"].astype(str)
splits = data["split"].astype(str)

X = X[..., np.newaxis]

# ================= LOAD GENRE LABELS =================

genre_labels = np.load(
    PROJECT_ROOT / "models" / "trained_models" / "genre_labels.npy",
    allow_pickle=False
)

label_to_index = {
    genre: index
    for index, genre in enumerate(genre_labels)
}

y = np.array([label_to_index[label] for label in labels])

# =================== TEST DATA =======================

test_mask = splits == "test"

X_test = X[test_mask]
y_test = y[test_mask]

print("Test samples:", len(X_test))

# ================ EVALUATE SAVED MODELS ===============

model_folder = PROJECT_ROOT / "models" / "trained_models"

results = read_mel_cnn_acc(
    model_folder,
    X_test,
    y_test,
    genre_labels
)

# =============== MODEL COMPARISON ===================

print("\n\n========== MODEL COMPARISON ==========")

results.sort(key=lambda x: x["accuracy"], reverse=True)

for result in results:

    print(
        f"{result['model']:<25} "
        f"Accuracy: {result['accuracy']:.2f}% | "
        f"Loss: {result['loss']:.4f}"
    )

