import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def split_data(x, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets while preserving
    the genre proportions in both sets.

    Parameters:
        x: Audio features.

        y: Target genre labels.

        test_size: Proportion of data reserved for testing.

        random_state: Seed for reproducible splits.

    Returns:
        x_train, x_test, y_train, y_test
    """
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

def split_by_song(x, y, groups, test_size=0.2, random_state=42):
    """
    Split audio segments into training and testing sets by song.
    Segments belonging to the same song remain in the same set.

    Parameters:
        x: Audio features.
    
        y: Target genre labels.

        groups: Song identifiers used to keep all segments from
        the same song in the same set.
    
        test_size (float): Proportion of data reserved for testing.
    
        random_state (int): Seed for reproducible splits.
    

    Returns:
        Training and testing features, labels, and group IDs.
    """

    group_split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(
        group_split.split(x, y, groups=groups)
    )

    return (
        x.iloc[train_idx],
        x.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        groups.iloc[train_idx],
        groups.iloc[test_idx]
    )

def scale_features(x_train, x_val=None, x_test=None):
    """
    Scale training and test data using StandardScaler.

    Parameters:
        x_train: Training features used to fit the scaler and transform.

        x_val: Optional validation features to transform. Defaults to None.

        x_test: Optional test features to transform. Defaults to None.

    Returns:
    Scaled features and scaler.
    """
    scaler = StandardScaler()

    x_train_sc = scaler.fit_transform(x_train)

    x_val_sc = (
        scaler.transform(x_val)
        if x_val is not None else None
    )

    x_test_sc = (
        scaler.transform(x_test)
        if x_test is not None else None
    )

    return x_train_sc, x_val_sc, x_test_sc, scaler

def build_model(shape=57, dense_1=64, dense_2=10):
    """
    Build and compile a fully connected neural network 
    for music genre classification.

    Parameters:
        shape (int): Number of input features.
        Defaults to 57.

        dense_1 (int): Number of neurons in the hidden Dense layer.
        Defaults to 64.

        dense_2 (int): Number of output units,
        typically one per genre class.
        Defaults to 10.

    Returns:
        A compiled Keras Sequential model using the Adam optimizer,
        sparse categorical cross-entropy loss, and accuracy metric.
    """

    model = models.Sequential([
    layers.Input(shape=(shape,)),
    layers.Dense(dense_1, activation="relu"),
    layers.Dense(dense_2, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def evaluate_model(model, x_test, y_test, label_encoder,
                   title="Model", plot=True):
    """
    Evaluate a trained classifier on the test set.

    Prints test loss, accuracy, and a classification report.
    Optionally displays a confusion matrix.

    Parameters:
        model: The trained neural network model
    
        x_test: The test dataset containing the input features.
    
        y_test: The true genre labels for the test dataset,
        encoded as integers.

        label_encoder: The LabelEncoder used to convert
        encoded genre labels back into their original names.

    Returns:
        A dictionary containing test loss, accuracy,
        predictions, and the confusion matrix.
    """
    loss, accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)

    class_ids = np.arange(len(label_encoder.classes_))

    print(f"\n{title} Test Accuracy: {accuracy * 100:.2f}%")
    print(f"{title} Test Loss: {loss:.4f}")

    print(f"\n{title} Classification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=class_ids,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=class_ids
    )

    if plot:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_encoder.classes_
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax)
        ax.set_title(f"{title} Confusion Matrix")
        ax.tick_params(axis="x", labelrotation=45)
        plt.tight_layout()
        plt.show()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": cm
    }
