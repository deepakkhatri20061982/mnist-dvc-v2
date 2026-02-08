# preprocess.py

import mlflow
import yaml
import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")



def main():
    with open("experiments/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    experiment_name = exp_cfg["experiment_name"]
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    params = yaml.safe_load(open("params.yaml"))["preprocess"]

    with mlflow.start_run(run_name="preprocess"):

        # Log params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Load data
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
        y = y.astype(int)

        mlflow.log_metric("total_samples", len(X))
        mlflow.log_metric("num_features", X.shape[1])

        mlflow.log_metric("pixel_min_before", X.min().min())
        mlflow.log_metric("pixel_max_before", X.max().max())

        if params["normalize"]:
            X = X / 255.0

        mlflow.log_metric("pixel_min_after", X.min().min())
        mlflow.log_metric("pixel_max_after", X.max().max())

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=params["test_size"],
            random_state=params["random_state"],
            stratify=y
        )

        # Save processed data (DVC tracks these)
        joblib.dump(X_train, "data/X_train.pkl")
        joblib.dump(X_test, "data/X_test.pkl")
        joblib.dump(y_train, "data/y_train.pkl")
        joblib.dump(y_test, "data/y_test.pkl")

        # Log small samples
        np.save("train_sample.npy", X_train.iloc[:100].values)
        mlflow.log_artifact("train_sample.npy")


if __name__ == "__main__":
    main()
