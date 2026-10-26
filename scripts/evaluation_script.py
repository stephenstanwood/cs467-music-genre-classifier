# Written by Dawson Herrington to automated creation of model evaluation metrics through scikit

# imports
from datetime import datetime
from pathlib import Path

import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# init wandb
run = wandb.init(project="music-genre-classifier", job_type="evaluation")


# load the saved model
model_path = Path(__file__).parent.parent / "models" / "model.keras"
model = tf.keras.models.load_model(model_path)

# load in the test split
csv_location = Path(__file__).parent.parent / "Data" / "gtzan_57_features_split.csv"
df = pd.read_csv(csv_location)
test_split = df[df["split"] == "test"].copy()

# apply same scaler and label as Gianno used
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

scaler = StandardScaler()
feature_columns = [col for col in df.columns if col not in ["filename", "song_id", "segment", "label", "split"]]
scaler.fit(train_df[feature_columns])
x_test_scaled = scaler.transform(test_df[feature_columns])

label_encoder = LabelEncoder()
label_encoder.fit(train_df["label"])
y_test_encoded = label_encoder.transform(test_df["label"])

# generate metrics
loss, accuracy = model.evaluate(x_test_scaled, y_test_encoded)
predictions = model.predict(x_test_scaled)
predicted_class_set =  predictions.argmax(axis=1)

accuracy = accuracy_score(y_test_encoded, predicted_class_set)

loss_acc_output_string = f"Loss: {loss}, Accuracy: {accuracy}"
output_matrix = confusion_matrix(y_test_encoded, predicted_class_set)

class_report = classification_report(y_test_encoded, predicted_class_set, target_names=label_encoder.classes_)

# log metrics to wandb
# After you have the metrics, log them
wandb.log({
    "test_loss": loss,
    "test_accuracy": accuracy,
})


# save to Evaluation Logs Folder
time = datetime.now().strftime("%Y-%m-%d_%I-%M%p")

folder = Path(__file__).parent.parent / "Evaluation Logs" / time
folder.mkdir()

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