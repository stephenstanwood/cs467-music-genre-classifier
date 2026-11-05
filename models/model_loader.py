import tensorflow as tf
import numpy as np
from pathlib import Path


"""
Load trained models from the trained_models folder and genre labels
from the genre_labels.npy file.

This script provides four reusable functions:
    * load_models()
    * get_active_models()
    * select_model(model_name)
    * predict_genre(X, model_name, top_n=3)

How to access:
    - from models.model_loader import *
    - when selecting a model, use only its name 
    without the .keras file extension.
"""


# ================= PROJECT PATH ======================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FOLDER = PROJECT_ROOT / "models" / "trained_models"


# ================= LOAD GENRE LABELS =================

GENRE_LABELS = np.load(
    MODEL_FOLDER / "genre_labels.npy",
    allow_pickle=False
)

# ================= AVAILABLE MODELS ==================

MODELS = {}

def load_models():
    """ 
    Load all saved Keras models from the trained models directory.

    Returns:
        A list of names of the loaded models.
    """

    MODELS.clear()

    for model_path in MODEL_FOLDER.glob("*.keras"):

        MODELS[model_path.stem] = tf.keras.models.load_model(
            model_path
        )

    print(f"Loaded {len(MODELS)} Trained Models.")

    return list(MODELS.keys())


# ================= GET AVAILABLE MODELS ==============

def get_active_models():
    """
    Retrieve the names of all currently loaded models.

    Returns:
        A list of available model names.
    """
    return list(MODELS.keys())


# ================= SELECT MODEL ======================

def select_model(model_name):
    """Retrieve a loaded model by its name.

    Parameters:
        - model_name: Name of the model.
    
    Returns:
        The selected keras model.
    """
    if model_name not in MODELS:
        raise ValueError("Model Not Found.")

    return MODELS[model_name]


# ================= PREDICT GENRE =====================

def predict_genre(X, model_name, top_n=3):
    """
    Generate top-N music genre predictions using a selected CNN model.

    Parameters:
        X (numpy.ndarray): Preprocessed mel spectrogram input.

        model_name (str): Name of the trained model to use.

        models (dict): Dictionary containing loaded Keras models.

        genre_labels (numpy.ndarray): Names of the genre classes.

        top_n (int): Number of top predictions to return.
        Defaults to 3.

    Returns:
        A list of dictionaries containing predicted genres
        and their confidence scores.

    References:
    Stack Overflow, Get top-N ranking indices and values, from numpy array
    """

    if model_name not in MODELS:
        raise ValueError("Model not found.")

    model = MODELS[model_name]

    if X.ndim == 3:
        X = np.expand_dims(X, axis=0)

    predictions = model.predict(X, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:top_n]

    results = []

    for index in top_indices:

        results.append({
            "genre": str(GENRE_LABELS[index]),
            "confidence": float(predictions[index])
        })

    return results

