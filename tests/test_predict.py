import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src import predict


@patch("src.predict.joblib.load")
@patch("src.predict.mlflow")
@patch("src.predict.yaml.safe_load")
def test_predict_main(
    mock_yaml,
    mock_mlflow,
    mock_joblib_load
):
    # ------------------------
    # Mock configs
    # ------------------------
    mock_yaml.side_effect = [
        {"experiment_name": "test-exp"},
        {"predict": {
            "num_samples": 5,
            "random_seed": 42
        }}
    ]

    # ------------------------
    # Fake data
    # ------------------------
    X_test = pd.DataFrame(np.random.rand(20, 10))
    y_test = pd.Series(np.random.randint(0, 2, 20))

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
    mock_model.predict_proba.return_value = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.2, 0.8],
        [0.9, 0.1]
    ])

    mock_joblib_load.side_effect = [
        X_test, y_test, mock_model
    ]

    # ------------------------
    # Run
    # ------------------------
    predict.main()

    # ------------------------
    # Assertions
    # ------------------------
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()
    mock_mlflow.log_metric.assert_called_once()
    mock_mlflow.log_artifact.assert_called_once()
