import mlflow
import mlflow.sklearn
import json

# Experiment name (must match Task 1)
experiment_name = "freshbasket-delivery-time-min"

# Get experiment
experiment = mlflow.get_experiment_by_name(experiment_name)

# Create client
client = mlflow.tracking.MlflowClient()

# Get all runs sorted by MAE (ascending)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.mae ASC"]
)

# Best run (lowest MAE)
best_run = runs[0]

run_id = best_run.info.run_id
mae = best_run.data.metrics["mae"]

# Model URI (IMPORTANT)
model_uri = f"runs:/{run_id}/model"

# Register model
result = mlflow.register_model(
    model_uri=model_uri,
    name="freshbasket-delivery-time-min-predictor"
)

# Prepare JSON output
output = {
    "registered_model_name": "freshbasket-delivery-time-min-predictor",
    "version": result.version,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": mae
}

# Save JSON
with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)
