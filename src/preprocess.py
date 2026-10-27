import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib, os
import re

# area converter: "5 Marla" / "2 Kanal" → float (normalise to Marla for consistency) 
def convert_area(area_str):
    match = re.match(r"(\d+\.?\d*)\s*(\w+)", str(area_str).lower())

    if not match:
        return np.nan

    value, unit = match.groups()
    value = float(value)

    conversions = {
        "marla": 1,
        "kanal": 20
    }

    return value * conversions.get(unit, np.nan)


def load_and_clean(path="data/raw/pakistan_properties.csv"):
    df = pd.read_csv(path)

    # dropping columns with no predictive value potential
    df = df.drop(columns=["property_id", "location_id", "page_url",
                           "latitude", "longitude", "date_added",
                           "agency", "agent", "province_name"])

    # keeping only the listings for sale
    df = df[df["purpose"] == "For Sale"].drop(columns=["purpose"])

    # converting area to numerical marlas
    df["area_marla"] = df["area"].apply(convert_area)
    df = df.drop(columns=["area"])

    # droping rows with missing/unrealistic values
    df = df.dropna()
    df = df[df["price"]      > 1_000_000]
    df = df[df["area_marla"] > 0]
    df = df[df["bedrooms"]   <= 10]
    df = df[df["baths"]      <= 10]

    # removing outliers
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[df["price"].between(lower, upper)]

    return df


def encode_and_split(df):
    df = df.copy()

    le_dict = {}
    for col in ["property_type", "location", "city"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(le_dict, "models/label_encoders.pkl")

    return X_train, X_test, y_train, y_test, le_dict


if __name__ == "__main__":
    df = load_and_clean()
    print("Clean shape:", df.shape)
    print(df.describe())

    X_train, X_test, y_train, y_test, _ = encode_and_split(df)
    print("\nTrain size:", X_train.shape)
    print("Test size: ", X_test.shape)
    print("\nFeatures:", X_train.columns.tolist())