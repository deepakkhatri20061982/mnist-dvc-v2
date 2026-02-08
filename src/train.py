# src/train.py

import os
import yaml
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

os.makedirs("model", exist_ok=True)

def main():
    with open("experiments/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    experiment_name = exp_cfg["experiment_name"]
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    params = yaml.safe_load(open("params.yaml"))["train"]

    X_train = joblib.load("data/X_train.pkl")
    X_test  = joblib.load("data/X_test.pkl")
    y_train = joblib.load("data/y_train.pkl")
    y_test  = joblib.load("data/y_test.pkl")

    with mlflow.start_run(run_name="train"):

        for k, v in params.items():
            mlflow.log_param(k, v)

        model = SGDClassifier(
            loss=params["loss"],
            max_iter=params["max_iter"],
            tol=params["tol"],
            random_state=params["random_state"],
            n_jobs=1   # important on Windows
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", acc)

        joblib.dump(model, "model/logreg_model.pkl")
        mlflow.sklearn.log_model(model, "model")

        print(f"Training completed | Accuracy: {acc:.4f}")

        mlflow.end_run(status="FINISHED")

if __name__ == "__main__":
    main()
