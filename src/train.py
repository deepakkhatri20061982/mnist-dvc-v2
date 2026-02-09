import os
import pickle
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import yaml
import joblib


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]


def main():
    os.makedirs("model", exist_ok=True)

    params = load_params()

    # =========================
    # Load preprocessed data
    # =========================
    X_train = joblib.load("data/X_train.pkl")
    y_train = joblib.load("data/y_train.pkl")
    X_test = joblib.load("data/X_test.pkl")
    y_test = joblib.load("data/y_test.pkl")

    classes = np.unique(y_train)

    # =========================
    # MLflow
    # =========================
    with open("experiments/experiment.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    experiment_name = exp_cfg["experiment_name"]
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="SGD_Epoch_Training"):

        mlflow.log_params(params)

        # =========================
        # Model
        # =========================
        model = SGDClassifier(
            loss=params["loss"],
            max_iter=1,              # IMPORTANT: 1 epoch per loop
            tol=None,                # Disable early stopping
            random_state=params["random_state"],
            learning_rate="optimal"
        )

        epochs = params["epochs"]

        print("\n🚀 Starting Training...\n")

        for epoch in range(1, epochs + 1):
            X_train, y_train = shuffle(X_train, y_train, random_state=epoch)

            if epoch == 1:
                model.partial_fit(X_train, y_train, classes=classes)
            else:
                model.partial_fit(X_train, y_train)

            # -------- Metrics --------
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)

            # Print epoch details
            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

            # Log per-epoch metrics
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)

        print("\n✅ Training Completed\n")

        # =========================
        # Final Evaluation
        # =========================
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, test_preds, average="weighted"
        )

        cm = confusion_matrix(y_test, test_preds)

        print("📊 Final Evaluation Metrics")
        print(f"Accuracy : {test_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # Log final metrics
        mlflow.log_metric("final_accuracy", test_acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save confusion matrix
        os.makedirs("artifacts", exist_ok=True)
        np.savetxt("artifacts/confusion_matrix.txt", cm, fmt="%d")
        mlflow.log_artifact("artifacts/confusion_matrix.txt")

        # =========================
        # Save & Log Model
        # =========================
        joblib.dump(model, "model/logreg_model.pkl")
        mlflow.sklearn.log_model(model, "model")

        print("📦 Model logged to MLflow")

        mlflow.end_run(status="FINISHED")


if __name__ == "__main__":
    main()
