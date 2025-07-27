import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Optional: Use XGBoost if installed
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False

def train_model(model_name="random_forest"):
    # Load scaled training data
    train_path = "data/features/combined_scaled_train.csv"
    test_path = "data/features/combined_scaled_test.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file '{train_path}' not found. Run scale mode first.")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None

    # 🔍 Check class balance in train/test data
    print("\n📊 Target class distribution in training data:")
    print(df_train["target"].value_counts())

    if df_test is not None:
        print("\n📊 Target class distribution in test data:")
        print(df_test["target"].value_counts())

    feature_cols = ["Close", "sentiment", "MA25", "MA50", "MACD", "Middle Band"]
    target_col = "target"

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if has_xgb:
        models["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: {list(models.keys())}")

    model = models[model_name]
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/stock_model_{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model '{model_name}' trained and saved to '{model_path}'")

    # Optional: Evaluate on test data if available
    if df_test is not None:
        X_test = df_test[feature_cols]
        y_test = df_test[target_col]
        y_pred = model.predict(X_test)

        print("\n📊 Evaluation on test data:")
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy:.4f}")

        # Save report to file with timestamp
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = f"results/test_report_{model_name}_{now}.txt"
        os.makedirs("results", exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Classification Report ({model_name}):\n")
            f.write(classification_report(y_test, y_pred))
            f.write(f"\nAccuracy: {accuracy:.4f}\n")
        print(f"📁 Report saved to '{report_path}'")

        # Plot accuracy as a bar chart
        plt.figure(figsize=(5, 4))
        plt.bar(["Accuracy"], [accuracy], color="skyblue")
        plt.ylim(0, 1)
        plt.title(f"Model Accuracy on Test Data ({model_name})")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        chart_path = f"results/test_accuracy_plot_{model_name}_{now}.png"
        plt.savefig(chart_path)
        plt.show()
        print(f"📊 Accuracy plot saved to '{chart_path}'")
