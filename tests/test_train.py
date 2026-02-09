import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src import train


@patch("src.train.joblib.dump")
@patch("src.train.joblib.load")
@patch("src.train.mlflow")
@patch("src.train.yaml.safe_load")
def test_train_main(
    mock_yaml_safe_load,
    mock_mlflow,
    mock_joblib_load,
    mock_joblib_dump
):
    # -------------------------
    # Mock YAML configs
    # -------------------------
    mock_yaml_safe_load.side_effect = [
        {
            "train": {
                "loss": "log_loss",
                "epochs": 2,
                "random_state": 42
            }
        },
        {
            "experiment_name": "test-exp"
        }
    ]

    # -------------------------
    # Fake but REAL data
    # -------------------------
    X_train = pd.DataFrame(np.random.rand(50, 10))
    y_train = pd.Series(np.random.randint(0, 2, 50))

    X_test = pd.DataFrame(np.random.rand(20, 10))
    y_test = pd.Series(np.random.randint(0, 2, 20))

    # -------------------------
    # Robust joblib.load mock
    # -------------------------
    def joblib_load_side_effect(path):
        if "X_train" in path:
            return X_train
        if "y_train" in path:
            return y_train
        if "X_test" in path:
            return X_test
        if "y_test" in path:
            return y_test
        raise ValueError(f"Unexpected joblib.load path: {path}")

    mock_joblib_load.side_effect = joblib_load_side_effect

    # -------------------------
    # Execute training
    # -------------------------
    train.main()

    # -------------------------
    # Assertions
    # -------------------------
    assert mock_joblib_load.call_count >= 4
    assert mock_joblib_dump.called

    mock_mlflow.set_experiment.assert_called_once_with("test-exp")
    mock_mlflow.start_run.assert_called_once()
