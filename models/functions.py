import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import (
    models, 
    layers, 
    regularizers)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay)


"""

functions.py

Contains reusable helper functions for music genre classification.

Functions include:

* data splitting
* audio segments splitting
* feature scaling
* model building
* training callbacks
* class weighting
* model evaluation

Dependencies:
    pandas
    numpy
    tensorflow
    matplotlib
    scikit-learn

"""


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

    Reference: Stack Overflow, python split to 
    train/test/val using GroupShuffleSplit
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


def build_model(shape=57, dense_1=64, dense_2=10, num_classes=10):
    """
    Build and compile a fully connected neural network 
    for music genre classification.

    Parameters:
        shape (int): Number of input features.
        Defaults to 57.

        dense_1 (int): Number of neurons in the
        first hidden layer.
        Defaults to 64.

        dense_2 (int): Number of neurons in the
        second hidden layer.
        Defaults to 10.

        num_classes (int): Number of genre classes to predict.
        Determines the number of neurons in the output layer.
        Defaults to 10.

    Returns:
        A compiled Keras Sequential model using the Adam optimizer,
        sparse categorical cross-entropy loss, and accuracy metric.

        # Settings:
           
                # L2 regularization penalizes large weights to help reduce overfitting.
                # Larger L2 values apply a stronger penalty.
                # L2 Value:
                # 0         No L2 Regularization
                # 0.0001    Weak
                # 0.001     Stronger
                # 0.01      Harshest
        
                # Randomly drops of activations during training.
                # 20% (0.2) for the first hidden layer
                # 10% (0.1) for the second hidden layer
        
                # Optimizer: Adam (Adaptive Moment Estimation)
                # Learning Rate:
                # 0.01          (unstable)
                # 0.001         (common)
                # 0.0001        (smaller updates)

    References: 
    Keras, Layer weight regularizers
    Keras, Model training APIs
    Stack Overflow, Setting the learning rate in convolutional neural network
    Stack Overflow, How to do Multiclass classification with Keras?
    """

    model = models.Sequential([

        # Input layer: receives the audio feature vector (57 features by default)
        layers.Input(shape=(shape,)),

        # First hidden layer.
        layers.Dense(
            dense_1, 
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),

        layers.Dropout(0.2),

        # Second hidden layer
        layers.Dense(
            dense_2, 
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.1),
     
        # Output layer: one neuron per genre class
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

    return model


def callbacks():
    """
    Initializes early stopping and learning rate reduction callbacks.
    Use the returned list in the model.fit() callbacks parameter.

    Returns:
        list: EarlyStopping, ReduceLROnPlateau
    """

    early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
    )

    return [early_stopping, reduce_lr]


def get_class_weights(labels_encoded):
    """
    Calculate class weights to help address imbalanced training data.

    Parameters:
        labels_encoded: Encoded genre labels encoded using LabelEncoder

    Returns:
        dict: Class labels mapped to their calculated weights.
    
    Reference:
    Stack Overflow, Compute class weight function issue in 'sklearn' 
    library when used in 'Keras' classification.
    """

    classes = np.unique(labels_encoded)

    weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels_encoded
    )

    return dict(zip(classes, weights))


def evaluate_model(model, x_test, y_test, label_encoder,
                   title="Model", plot=True):
    """
    Evaluate a trained classifier on the test set.

    Prints test loss, accuracy, and a classification report.
    Optionally displays a confusion matrix.

    Parameters:
        model: The trained neural network model.
    
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

