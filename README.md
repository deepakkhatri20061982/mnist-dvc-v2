# DVC Initialization
dvc init

# DVC Commands
dvc stage add -n preprocess -d src/preprocess.py -d data/raw -d params.yaml -o data/processed python src/preprocess.py

dvc stage add -n train -d src/train.py -d src/model.py -d data/processed -d params.yaml -o model.pt -M metrics.json python src/train.py


# To run using DVC
dvc repro

dvc repro create_experiment --force --downstream

dvc repro --force

# Check Metrics
dvc metrics show

dvc metrics diff

dvc params diff


# Check Experiments
dvc exp show

dvc exp diff
