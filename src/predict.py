# predict.py

import yaml
import joblib
import numpy as np
import mlflow
import pandas as pd


def main():
    with open("experiments/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    experiment_name = exp_cfg["experiment_name"]

    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    params = yaml.safe_load(open("params.yaml"))["predict"]
    np.random.seed(params["random_seed"])

    # -------------------------
    # Load data
    # -------------------------
    X_test = joblib.load("data/X_test.pkl")
    y_test = joblib.load("data/y_test.pkl")

    # -------------------------
    # Load model LOCALLY ✅
    # -------------------------
    model = joblib.load("model/logreg_model.pkl")

    with mlflow.start_run(run_name="predict"):

        mlflow.log_param("num_samples", params["num_samples"])
        mlflow.log_param("random_seed", params["random_seed"])

        indices = np.random.choice(
            len(X_test),
            params["num_samples"],
            replace=False
        )

        X_sample = X_test.iloc[indices]
        y_true = y_test.iloc[indices]

        y_pred = model.predict(X_sample)
        y_proba = model.predict_proba(X_sample)
        confidence = np.max(y_proba, axis=1)

        df_preds = pd.DataFrame({
            "index": indices,
            "true_label": y_true.values,
            "predicted_label": y_pred,
            "confidence": confidence
        })

        sample_accuracy = (y_true.values == y_pred).mean()
        mlflow.log_metric("sample_accuracy", sample_accuracy)

        df_preds.to_csv("prediction_samples.csv", index=False)
        mlflow.log_artifact("prediction_samples.csv")

        print(df_preds)
        print(f"\nSample Accuracy: {sample_accuracy:.4f}")


if __name__ == "__main__":
    main()
