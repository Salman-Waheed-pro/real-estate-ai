import pandas as pd
import numpy as np
import joblib
from src.preprocess import load_and_clean, convert_area

def load_models():
    rf = joblib.load("models/rf_model.pkl")
    le_dict = joblib.load("models/label_encoders.pkl")
    df = load_and_clean()
    return rf, le_dict, df


def encode_input(row: dict, le_dict: dict) -> dict:
    encoded = row.copy()
    for col, le in le_dict.items():
        val = str(row.get(col, ""))
        if val in le.classes_:
            encoded[col] = le.transform([val])[0]
        else:
            encoded[col] = most_common = le.classes_[0]
    return encoded


def predict_price(model, le_dict, property_type, location, city,
                  baths, bedrooms, area_marla) -> float:
    row = {
        "property_type": property_type,
        "location": location,
        "city": city,
        "baths": baths,
        "bedrooms": bedrooms,
        "area_marla": area_marla
    }
    encoded = encode_input(row, le_dict)
    X = pd.DataFrame([encoded])
    return float(model.predict(X)[0])


def score_property(row, budget):
    score = 0

    gap = budget - row["price"]
    if gap >= 0:
        score += min(gap / budget, 1.0) * 40

    gap2 = row["predicted_price"] - row["price"]
    if gap2 > 0:
        score += min(gap2 / row["price"], 0.5) * 40

    score += min(row["area_marla"] / 20, 1.0) * 10

    score += min(row["bedrooms"] / 5, 1.0) * 10

    return round(score, 2)


def recommend(budget, city, property_type,
              min_beds=1, min_baths=1, min_area=2,
              top_n=5):

    rf, le_dict, df = load_models()

    mask = (
        (df["city"] == city) &
        (df["property_type"] == property_type) &
        (df["price"] <= budget) &
        (df["price"] >= 1_000_000) &
        (df["bedrooms"] >= min_beds) &
        (df["baths"] >= min_baths) &
        (df["area_marla"] >= min_area)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        return pd.DataFrame(), "No properties found matching your criteria."

    batch = filtered.copy()
    for col, le in le_dict.items():
        batch[col] = batch[col].astype(str).map(
            lambda v, le=le: le.transform([v])[0] if v in le.classes_ else 0
        )

    feature_cols = ["property_type", "location", "city", "baths", "bedrooms", "area_marla"]
    filtered["predicted_price"] = rf.predict(batch[feature_cols])

    filtered["investment_score"] = filtered.apply(
        lambda r: score_property(r, budget), axis=1
    )

    cols = ["city", "location", "property_type",
            "bedrooms", "baths", "area_marla",
            "price", "predicted_price", "investment_score"]

    result = (filtered[cols]
              .sort_values("investment_score", ascending=False)
              .head(top_n)
              .reset_index(drop=True))

    return result, "success"


if __name__ == "__main__":
    results, status = recommend(
        budget = 15_000_000,
        city = "Lahore",
        property_type = "House",
        min_beds = 3,
        min_baths = 2,
        min_area = 5,
        top_n = 5
    )
    if status == "success":
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(results.to_string(index=False))
    else:
        print(status)