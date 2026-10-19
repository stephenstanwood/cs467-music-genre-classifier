import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay)


# BASELINE MODEL
# References: Kaavya Mahajan, "Training a TensorFlow model 
# for predicting the Genres of Popular Songs"

# ==================== LOAD AND PREPARE DATA ====================
df = pd.read_csv(
    "Data/" \
    "features_30_sec.csv"
    )

x = df.drop(columns=["filename", "label", "length"])
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================== SCALE FEATURES/ ENCODE LABELS ===============
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# ===================== NEURAL NETWORK ===========================
# Build Model
# Input: 57 Audio Features
# 64 Neurons, ReLU: Rectified Linear Unit
# softmax: Converts scores into probabilities totaling 1

model = models.Sequential([
    layers.Input(shape=(57,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
# epochs=30: Passes through training data up to 30 times
# batch_size=32: Processes 32 training examples per batch
# validation_split=0.2: Reserves 20% of training data for validation

history = model.fit(
    x_train_scaled,
    y_train_encoded,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# ====================== EVALUATE MODEL ===========================
test_loss, test_accuracy = model.evaluate(
    x_test_scaled,
    y_test_encoded,
    verbose=0
)

print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))

# Generate Predictions
preds_test = model.predict(x_test_scaled, verbose=0)
preds_test = preds_test.argmax(axis=1)

# Classification Report
print(classification_report(
    y_test_encoded,
    preds_test,
    target_names=label_encoder.classes_
))

# Confusion Matrix
cm = confusion_matrix(
    y_test_encoded,
    preds_test
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

# Plot Matrix
fig, ax = plt.subplots(figsize=(8,8))

disp.plot(ax=ax)

ax.tick_params(axis="x", labelsize=10, rotation=45)
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()

plt.show()

