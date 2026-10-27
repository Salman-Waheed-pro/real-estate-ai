import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import callback
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import load_and_clean, encode_and_split
from train import evaluate, plot_predictions, plot_feature_importance


def train_xgb(X_train, X_test, y_train, y_test):
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )

    y_pred  = xgb.predict(X_test)
    metrics = evaluate(y_test, y_pred, label="XGBoost")

    plot_predictions(y_test.values, y_pred, label="XGBoost")
    plot_feature_importance(xgb, X_train.columns.tolist())

    joblib.dump(xgb, "models/xgb_model.pkl")
    print("\nModel saved → models/xgb_model.pkl")
    return xgb, metrics


def compare_models(rf_metrics, xgb_metrics):
    labels  = ["MAE (PKR)", "RMSE (PKR)", "R²"]
    rf_vals  = [rf_metrics["mae"],  rf_metrics["rmse"],  rf_metrics["r2"]]
    xgb_vals = [xgb_metrics["mae"], xgb_metrics["rmse"], xgb_metrics["r2"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for i, (ax, label, rv, xv) in enumerate(zip(axes, labels, rf_vals, xgb_vals)):
        bars = ax.bar(["Random Forest", "XGBoost"], [rv, xv],
                      color=["purple", "orange"], alpha=0.85, width=0.5)
        ax.set_title(label)
        ax.set_ylabel(label)

        # value labels on top of each bar
        for bar, val in zip(bars, [rv, xv]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:,.4f}" if label == "R²" else f"{val:,.0f}",
                    ha="center", va="bottom", fontsize=9)

    plt.suptitle("Random Forest vs XGBoost - Performance Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig("models/model_comparison.png", dpi=150)
    plt.show()

    print("\n--- Comparison Summary ---")
    print(f"{'Metric':<10} {'Random Forest':>15} {'XGBoost':>15} {'Winner':>10}")
    print("-" * 55)
    print(f"{'MAE':<10} {rf_metrics['mae']:>15,.0f} {xgb_metrics['mae']:>15,.0f} "
          f"{'RF' if rf_metrics['mae'] < xgb_metrics['mae'] else 'XGB':>10}")
    print(f"{'RMSE':<10} {rf_metrics['rmse']:>15,.0f} {xgb_metrics['rmse']:>15,.0f} "
          f"{'RF' if rf_metrics['rmse'] < xgb_metrics['rmse'] else 'XGB':>10}")
    print(f"{'R²':<10} {rf_metrics['r2']:>15.4f} {xgb_metrics['r2']:>15.4f} "
          f"{'RF' if rf_metrics['r2'] > xgb_metrics['r2'] else 'XGB':>10}")


if __name__ == "__main__":
    df                                   = load_and_clean()
    X_train, X_test, y_train, y_test, _  = encode_and_split(df)

    rf = joblib.load("models/rf_model.pkl")
    rf_preds    = rf.predict(X_test)
    rf_metrics  = {
        "mae":  mean_absolute_error(y_test, rf_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "r2":   r2_score(y_test, rf_preds)
    }

    xgb, xgb_metrics = train_xgb(X_train, X_test, y_train, y_test)
    compare_models(rf_metrics, xgb_metrics)