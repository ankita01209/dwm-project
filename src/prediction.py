"""
src/prediction.py — Model loading for CarWise.

Loads models/car_price_random_forest.pkl via joblib.
Returns (model, error_message). If model is None, error_message explains why.
Never retrains, never falls back to an alternative model.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = "models/car_price_random_forest.pkl"

# Expected feature order — must match the training pipeline exactly.
PREDICTION_FEATURES = [
    "car_age",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "brand",
]


def load_model():
    """
    Load the trained Random Forest pipeline from disk.

    Returns:
        (model, None)  — on success
        (None, str)    — on failure; str contains a user-facing error message
    """
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file not found: `{MODEL_PATH}`. "
            "Ensure the trained pipeline has been placed in the models/ directory."
        )

    try:
        model = joblib.load(MODEL_PATH)
        return model, None

    except AttributeError as e:
        # Most common cause: scikit-learn version mismatch between the
        # Colab environment where the model was trained and the local venv.
        try:
            import sklearn
            local_sklearn = sklearn.__version__
        except ImportError:
            local_sklearn = "unknown"

        try:
            local_joblib = joblib.__version__
        except Exception:
            local_joblib = "unknown"

        return None, (
            "**Model loading failed — scikit-learn version mismatch.**\n\n"
            f"Your local environment has **scikit-learn {local_sklearn}** and "
            f"**joblib {local_joblib}**, which are incompatible with the saved pkl.\n\n"
            "**To fix this**, please install the compatible scikit-learn version it was saved with:\n"
            "```\n"
            "pip install scikit-learn==1.6.1 joblib\n"
            "```\n"
            "Or run `pip install -r requirements.txt` if available.\n\n"
            f"**Raw error:** `{e}`"
        )

    except Exception as e:
        return None, (
            f"**Model loading failed.**\n\n"
            f"Error: `{type(e).__name__}: {e}`\n\n"
            "Check that `models/car_price_random_forest.pkl` is a valid joblib file "
            "and that your Python/scikit-learn environment matches the training environment."
        )


def predict(model, car_age: int, km_driven: int, fuel: str,
            seller_type: str, transmission: str, owner: str, brand: str) -> float:
    """
    Run inference using the loaded pipeline.
    car_age is computed by the caller as: 2020 - year

    Returns the predicted selling price (float).
    Raises ValueError or RuntimeError on failure.
    """
    input_df = pd.DataFrame([{
        "car_age": car_age,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
    }])[PREDICTION_FEATURES]  # enforce column order

    prediction = model.predict(input_df)[0]
    return float(max(0.0, prediction))
