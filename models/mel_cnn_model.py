
from functions import *

"""
Training the Model Using GTZAN Mel Spectrograms (3-Second Segments)

Pipeline:
1. Load preprocessed Mel spectrograms.

2. Encode genre labels.

3. Use predefined song-level training, validation, and test splits.

4. Verify no song-level data leakage.

5. Build and train Mel CNN V2.

6. Evaluate validation performance.

7. Save model labels.

Test evaluation should only happen after final model selection.
"""

# ====================== SET SEED ======================

tf.keras.utils.set_random_seed(42)

# ==================== LOAD DATA =======================

data = np.load("Data/gtzan_mels.npz", allow_pickle=True)

X = data["X"].astype(np.float32)
labels = data["label"].astype(str)
splits = data["split"].astype(str)
song_ids = data["song_id"].astype(str)

# Add channel dimension
X = X[..., np.newaxis]

# ================== ENCODE LABELS =====================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# ==================== SPLIT DATA ======================

train_mask = splits == "train"
val_mask = np.isin(splits, ["val", "validation"])
test_mask = splits == "test"

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# ================= DATA LEAKAGE CHECK =================

train_songs = set(song_ids[train_mask])
val_songs = set(song_ids[val_mask])
test_songs = set(song_ids[test_mask])

assert train_songs.isdisjoint(val_songs)
assert train_songs.isdisjoint(test_songs)
assert val_songs.isdisjoint(test_songs)

print("Song-level split check passed!")

# ==================== DATA CHECKS =====================

assert len(X_train) > 0
assert len(X_val) > 0
assert len(X_test) > 0

assert X_train.shape[1:] == (128, 130, 1)
assert np.isfinite(X_train).all()
assert np.isfinite(X_val).all()

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# ===================== NEURAL NETWORK =================

model = build_mel_cnn_v2(
    num_classes=len(label_encoder.classes_)
)

model.summary()

# ===================== TRAIN MODEL ====================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks_v2(),
    verbose=1
)

# ================= VALIDATION RESULTS =================

results = evaluate_model(
    model=model,
    x_test=X_val,
    y_test=y_val,
    label_encoder=label_encoder,
    title="Mel CNN V2 - Validation",
    plot=True
)

# ==================== BEST EPOCH ======================

best_epoch = np.argmin(history.history["val_loss"])

print("\nBest Epoch:", best_epoch + 1)

print("Validation Accuracy at Best Epoch:",
      history.history["val_accuracy"][best_epoch])

print("Best Validation Loss:",
      history.history["val_loss"][best_epoch])

# ===================== SAVE LABELS ====================

np.save("genre_labels.npy", label_encoder.classes_)

# ==================== DATA SUMMARY ====================

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("\nMel CNN V2 training complete!")

np.save(
    "models/trained_models/genre_labels.npy",
    label_encoder.classes_
)

# Make active to save New Model
model.save("mel_cnn_v9.keras")

# Make active before saving new models
print("Model saved successfully!")
