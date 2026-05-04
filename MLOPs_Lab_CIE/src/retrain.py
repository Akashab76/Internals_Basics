import pandas as pd
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

# Load original data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

# ORIGINAL split (IMPORTANT: SAME as Task 1)
X = train_df.drop("delivery_time_min", axis=1)
y = train_df["delivery_time_min"]

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Combine datasets
combined_df = pd.concat([train_df, new_df])

X_comb = combined_df.drop("delivery_time_min", axis=1)
y_comb = combined_df["delivery_time_min"]

# Retrain SAME MODEL TYPE (Lasso)
model = Lasso()
model.fit(X_comb, y_comb)

# Evaluate on SAME test set
preds = model.predict(X_test_orig)
retrained_mae = mean_absolute_error(y_test_orig, preds)

# Champion MAE (from Task 1)
champion_mae = 3.744422791709053

# Improvement
improvement = champion_mae - retrained_mae

# Threshold
threshold = 1.0

if improvement >= threshold:
    action = "promoted"
else:
    action = "kept_champion"

# Output JSON
output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "mae"
}

# Save
with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)
