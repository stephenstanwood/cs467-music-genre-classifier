# Course: CS 467 - Online Capstone Project
# Project: Top-n Music Genre Classification Neural Network
# Team: Sam Davidson, Dawson Ross Herrington, Ashcon Khoubyari,
#       Gianno Mari Rodriguez, Stephen Stanwood
# Description: Compares a pre-trained model (YAMNet) against our model on the same split.

# imports
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
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

# init wandb
run = wandb.init(project="music-genre-classifier", job_type="yamnet_comparison")

# load the pre-trained model
yamnet = tf.saved_model.load("/tmp/yamnet")

# load in the appropriate split, one row per song
csv_location = Path(__file__).parent.parent / "Data" / "gtzan_57_features_split.csv"
df = pd.read_csv(csv_location)
songs = df[df["split"].isin(["train", "val"])].groupby("song_id")[["label", "split"]].first()

audio_folder = Path("/tmp/gtzan/Data/genres_original")

# one averaged yamnet embedding per song
embeddings, labels, splits = [], [], []

# go through each song one at a time
# n counts songs for the progress printout
for n, (song_id, row) in enumerate(songs.iterrows(), 1):
    # load the song's audio, yamnet requires 16kHz mono audio
    try:
        waveform, _ = librosa.load(audio_folder / row["label"] / f"{song_id}.wav", sr=16000, mono=True)
    # skip any song that can't be read (e.g. corrupted files) instead of crashing
    except Exception as e:
        print(f"skipped {song_id}: {e}")
        continue
    # run the audio through yamnet
    # yamnet returns 3 things: sound scores, embeddings, and its spectrogram - we only keep the embeddings
    # embeddings are 1024 numbers describing what each ~1 second frame sounds like
    _, song_embeddings, _ = yamnet(waveform.astype(np.float32))
    # average every frame together so each song gets one set of 1024 numbers
    embeddings.append(song_embeddings.numpy().mean(axis=0))
    # keep the genre and train/val split lined up with each song's embedding
    labels.append(row["label"])
    splits.append(row["split"])
    # print progress every 50 songs
    if n % 50 == 0:
        print(f"  {n}/{len(songs)} songs done")

# convert the lists to numpy arrays so we can filter them by split
embeddings = np.array(embeddings)
labels = np.array(labels)
splits = np.array(splits)

# train a small classifier on the train songs
scaler = StandardScaler()
x_train = scaler.fit_transform(embeddings[splits == "train"])
x_val = scaler.transform(embeddings[splits == "val"])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(labels[splits == "train"])
y_test_encoded = label_encoder.transform(labels[splits == "val"])

classifier = LogisticRegression(max_iter=1000)
classifier.fit(x_train, y_train)

# generate metrics
predictions = classifier.predict_proba(x_val)
predicted_class_set = predictions.argmax(axis=1)

loss = log_loss(y_test_encoded, predictions)
accuracy = accuracy_score(y_test_encoded, predicted_class_set)

loss_acc_output_string = f"YAMNet + LogisticRegression (song-level, val split)\nLoss: {loss}, Accuracy: {accuracy}"
output_matrix = confusion_matrix(y_test_encoded, predicted_class_set)

class_report = classification_report(y_test_encoded, predicted_class_set, target_names=label_encoder.classes_)

# log metrics to wandb
wandb.log({
    "yamnet_loss": loss,
    "yamnet_accuracy": accuracy,
})

# create the parent folder and keep each evaluation's results separate.
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
folder = Path(__file__).parent.parent / "Evaluation Logs" / f"{time}_yamnet"
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
completed_matrix = ConfusionMatrixDisplay(output_matrix, display_labels=label_encoder.classes_)

completed_matrix.plot()

# adjust x-axis for space
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(folder / "confusion_matrix.png")

# finish wandb
wandb.finish()