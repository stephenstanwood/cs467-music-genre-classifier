# Written by Dawson Herrington to automated creation of model evaluation metrics through scikit

# imports
from datetime import datetime
from pathlib import Path
from pyexpat import model

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

# load the saved model
model_path = Path(__file__).parent.parent / "models" / "model.keras"

# load in the test split
csv_location = Path(__file__).parent.parent / "Data" / "gtzan_57_features_split.csv"
df = pd.read_csv(csv_location)
test_split = df[df["split"] == "test"].copy()

# apply same scaler and label as Gianno used
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

scaler = StandardScaler()
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

# save to Evaluation Logs Folder
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

folder = Path(__file__).parent.parent / "Evaluation_Logs"
folder.mkdir()


