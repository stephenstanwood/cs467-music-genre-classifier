from functions import *


"""
Training the Model Using GTZAN Mel Spectrograms (3-Second Segments)

Pipeline:

1. Load and Prepare Data: Load the preprocessed GTZAN mel spectrogram
   dataset (gtzan_mels.npz). Convert the input data to float32 and
   expand its dimensions to include a channel axis for CNN input
   using np.newaxis.

2. Encode Labels: Convert the genre labels into numerical values.

3. Split Data: Use the predefined training, validation, and testing
   splits from the dataset. Verify that song IDs are mutually
   exclusive across all three sets to prevent data leakage.

4. Build and Train the Model: Build a Convolutional Neural Network
   (CNN) designed to learn patterns from mel spectrograms. Train
   for up to 100 epochs with a batch size of 32, using validation
   data and callbacks to monitor and optimize training.

5. Evaluate the Model: Evaluate performance on the unseen test set
   using accuracy, loss, classification metrics, and diagnostic
   plots to assess model performance and identify areas for
   improvement.

6. Save Model Labels: Save the trained model's genre labels as a
   NumPy file (genre_labels.npy) for consistent label mapping
   during future predictions.

------------------------------------------------------------------

Training and Validation Performance Analysis of the Model:

GTZAN MEL SPECTROGRAMS:
Dataset: gtzan_mels.npz

* Test Accuracy: 78.00% - 80.40%
"""


# Set seed
tf.keras.utils.set_random_seed(42)

# ==================== LOAD AND PREPARE DATA ====================

data = np.load("Data/gtzan_mels.npz", allow_pickle=True)

X = data["X"].astype(np.float32)
labels = data["label"].astype(str)
splits = data["split"].astype(str)
song_ids = data["song_id"].astype(str)

# Add channel axis for CNN
X = X[..., np.newaxis]

# ================== SCALE FEATURES/ ENCODE LABELS ===============

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

train_mask = splits == "train"
val_mask = np.isin(splits, ["val", "validation"])
test_mask = splits == "test"

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Check
train_songs = set(song_ids[train_mask])
val_songs = set(song_ids[val_mask])
test_songs = set(song_ids[test_mask])

assert train_songs.isdisjoint(val_songs)
assert train_songs.isdisjoint(test_songs)
assert val_songs.isdisjoint(test_songs)

print("Song-level split check passed!")

# ===================== NEURAL NETWORK ===========================

model = build_mel_cnn(
    num_classes=len(label_encoder.classes_)
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks(),
    verbose=1
)

# ====================== EVALUATE MODEL ===========================

results = evaluate_model(
    model=model,
    x_test=X_test,
    y_test=y_test,
    label_encoder=label_encoder,
    title="Mel Spectrogram CNN",
    plot=True
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("\nBest validation accuracy:",
      max(history.history["val_accuracy"]))

print("Best validation loss:",
      min(history.history["val_loss"]))

"""
##################### SAVED MODELS #######################
#                                                        #
#     Here are the saved models, all trained, tested,    #
#        and validated using the GTZAN dataset.          #
#     Future models may use different datasets, so this  # 
#        section will be updated accordingly.            #
#                                                        #
##########################################################

# 79%
# model.save("mel_cnn.keras")

# 78%
# model.save("mel_cnn_v1.keras")

# 80.10%
# model.save("mel_cnn_v2.keras")

# 80.40%
# model.save("mel_cnn_v3.keras")

# 78.90%
# model.save("mel_cnn_v4.keras")

# 80.00%
# model.save("mel_cnn_v5.keras")

# 80.40%
# model.save("mel_cnn_v6.keras")

##########################################################

np.save(
    "models/trained_models/genre_labels.npy",
    label_encoder.classes_
)
"""

# Make active to save New Model
# model.save("...")

# Make active before saving new models
# print("Model saved successfully!")

