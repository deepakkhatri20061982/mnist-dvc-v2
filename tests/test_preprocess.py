import numpy as np
import pandas as pd
from unittest.mock import patch

from src import preprocess


@patch("src.preprocess.mlflow")
@patch("src.preprocess.joblib.dump")
@patch("src.preprocess.joblib.load")
@patch("src.preprocess.yaml.safe_load")
def test_preprocess_main(
    mock_yaml_safe_load,
    mock_joblib_load,
    mock_joblib_dump,
    mock_mlflow
):
    # --------------------------------------------------
    # Robust YAML mocking (path-based, not order-based)
    # --------------------------------------------------
    def yaml_side_effect(file_obj):
        name = getattr(file_obj, "name", "")

        if "params" in name:
            return {
                "preprocess": {
                    "test_size": 0.2,
                    "random_state": 42,
                    "normalize": True
                }
            }

        if "config" in name or "experiment" in name:
            return {
                "experiment_name": "test-exp"
            }

        if "schema" in name:
            return {}

        raise ValueError(f"Unexpected YAML file: {name}")

    mock_yaml_safe_load.side_effect = yaml_side_effect

    # --------------------------------------------------
    # Fake raw dataset
    # --------------------------------------------------
    X = pd.DataFrame(np.random.rand(100, 784))
    y = pd.Series(np.random.randint(0, 10, 100))

    mock_joblib_load.side_effect = [X, y]

    # --------------------------------------------------
    # Execute preprocessing
    # --------------------------------------------------
    preprocess.main()

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------
    mock_mlflow.set_experiment.assert_called_once_with("test-exp")
    mock_mlflow.start_run.assert_called_once()
    assert mock_joblib_dump.call_count >= 4
