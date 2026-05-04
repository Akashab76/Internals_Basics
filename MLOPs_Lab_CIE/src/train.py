import pandas as pd
import mlflow
import mlflow.sklearn
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("delivery_time_min", axis=1)
y = df["delivery_time_min"]

# Split (STRICT from PDF)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("freshbasket-delivery-time-min")

models = {
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log params + metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"mae": mae, "rmse": rmse})

        # REQUIRED TAG
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # SAVE MODEL (IMPORTANT for models/)
        joblib.dump(model, f"models/{name}.pkl")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

# BEST MODEL BY MAE (IMPORTANT)
best_model = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "freshbasket-delivery-time-min",
    "models": results,
    "best_model": best_model["name"],
    "best_metric_name": "mae",
    "best_metric_value": best_model["mae"]
}

# Save JSON
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)
