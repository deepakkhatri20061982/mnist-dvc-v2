# train.py

import yaml
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


def main():
    with open("experiments/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    experiment_name = exp_cfg["experiment_name"]

    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    params = yaml.safe_load(open("params.yaml"))["train"]

    # Load processed data
    X_train = joblib.load("data/X_train.pkl")
    X_test = joblib.load("data/X_test.pkl")
    y_train = joblib.load("data/y_train.pkl")
    y_test = joblib.load("data/y_test.pkl")

    with mlflow.start_run(run_name="train"):

        # Log hyperparams
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Metrics
        mlflow.log_metric("iterations_used", int(model.n_iter_[0]))

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        np.save("confusion_matrix.npy", cm)
        mlflow.log_artifact("confusion_matrix.npy")

        # Classification report
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("classification_report.txt")

        # Log model
        # mlflow.sklearn.log_model(
        #     model,
        #     artifact_path="model",
        #     registered_model_name="MNIST_LogisticRegressionV2"
        # )

        # Save model locally (DVC-tracked)
        joblib.dump(model, "model/logreg_model.pkl")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print(f"Training completed | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
