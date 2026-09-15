import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# Load trained classifier and supporting files
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

model = joblib.load(
    MODEL_DIR / "credit_card_classification_model.pkl"
)

encoder = joblib.load(
    MODEL_DIR / "product_type_label_encoder.pkl"
)

feature_columns = joblib.load(
    MODEL_DIR / "classification_feature_columns.pkl"
)


# ============================================================
# Predict Product Identity
# ============================================================

def predict_product_identity(product_data: dict) -> str:

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in product_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing classification features: {missing_features}"
        )

    input_df = pd.DataFrame([product_data])

    input_df = input_df.reindex(
        columns=feature_columns
    )

    for column in feature_columns:
        input_df[column] = pd.to_numeric(
            input_df[column],
            errors="raise"
        )

    encoded_prediction = model.predict(
        input_df
    ).astype(int)

    predicted_identity = encoder.inverse_transform(
        encoded_prediction
    )[0]

    return predicted_identity
