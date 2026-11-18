# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Evaluates a saved model and records its results.

# Written by Dawson Herrington to automated creation of model evaluation metrics through scikit

# imports
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# init wandb
run = wandb.init(project="music-genre-classifier", job_type="evaluation")


# load the saved model
#model_path = Path(__file__).parent.parent / "models" / "model.keras"
#model = tf.keras.models.load_model(model_path)

# load the saved model
model_path = Path(__file__).parent.parent / "models" / "trained_models" / "mel_cnn.keras"
model = tf.keras.models.load_model(model_path)
genre_labels = list(np.load(model_path.parent / "genre_labels.npy"))


# load in the appropriate split
mels_location = Path(__file__).parent.parent / "Data" / "gtzan_mels.npz"
data = np.load(mels_location)
#test_split = df[df["split"] == "test"].copy()
test_split = data["split"] == "val"

X_val = data["X"][test_split][..., np.newaxis].astype(np.float32)
labels_val = data["label"][test_split]
songs_val = data["song_id"][test_split]

segment_predictions = model.predict(X_val, verbose=0)

y_true = []
song_predictions = []

for song_id in np.unique(songs_val):
    song_rows = songs_val == song_id
    song_predictions.append(np.mean(segment_predictions[song_rows], axis=0))
    y_true.append(genre_labels.index(labels_val[song_rows][0]))

y_test_encoded = np.array(y_true)
predictions = np.array(song_predictions)

# generate metrics
loss = log_loss(y_test_encoded, predictions)
predicted_class_set =  predictions.argmax(axis=1)

accuracy = accuracy_score(y_test_encoded, predicted_class_set)

loss_acc_output_string = f"Loss: {loss}, Accuracy: {accuracy}"
output_matrix = confusion_matrix(y_test_encoded, predicted_class_set)

class_report = classification_report(y_test_encoded, predicted_class_set, target_names=genre_labels)

# log metrics to wandb
# After you have the metrics, log them
wandb.log({
    "test_loss": loss,
    "test_accuracy": accuracy,
})


# create the parent folder and keep each evaluation's results separate.
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
folder = Path(__file__).parent.parent / "Evaluation Logs" / time
folder.mkdir(parents=True)

# add report txt file into folder
with open(folder / "report.txt", "w") as f:
    f.write(loss_acc_output_string + "\n\n")
    f.write(class_report)

print(run.url)
# link file
with open(folder / "wandb_link.txt", "w") as f:
    f.write(run.url)

# generate confusion matrix .png to folder
completed_matrix = ConfusionMatrixDisplay(output_matrix, display_labels=genre_labels)

completed_matrix.plot()

# adjust x-axis for space
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(folder / "confusion_matrix.png")

# finish wandb
wandb.finish()