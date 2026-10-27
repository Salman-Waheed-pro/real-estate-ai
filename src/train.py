import numpy as np
import matplotlib.pyplot as plt
import joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import load_and_clean, encode_and_split

def evaluate(y_test, y_pred, label="Model"):
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"\n{label}")
    print(f"  MAE  : PKR {mae:,.0f}")
    print(f"  RMSE : PKR {rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_predictions(y_test, y_pred, label="Random Forest"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="purple", edgecolors="none", s=10)

    # prediction line
    mn, mx = y_test.min(), y_test.max()
    plt.plot([mn, mx], [mn, mx], color="orange", linewidth=1.5, label="Perfect prediction")

    plt.xlabel("Actual Price (PKR)")
    plt.ylabel("Predicted Price (PKR)")
    plt.title(f"{label} : Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"models/{label.lower().replace(' ', '_')}_predictions.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)),
            importances[indices],
            color="purple", alpha=0.8)
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in indices],
               rotation=45, ha="right")
    plt.title("Feature Importance — Random Forest")
    plt.tight_layout()
    plt.savefig("models/rf_feature_importance.png", dpi=150)
    plt.show()


def train_rf(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    metrics = evaluate(y_test, y_pred, label="Random Forest")
    plot_predictions(y_test.values, y_pred)
    plot_feature_importance(rf, X_train.columns.tolist())

    joblib.dump(rf, "models/rf_model.pkl")
    print("\nModel saved → models/rf_model.pkl")
    return rf, metrics


if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test, _ = encode_and_split(df)
    rf, metrics = train_rf(X_train, X_test, y_train, y_test)