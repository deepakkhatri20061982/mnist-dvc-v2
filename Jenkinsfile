pipeline {
    agent any
    
    stages {
        stage("Install Dependencies") {
            steps {
                sh """
                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }
        stage('Verify Python Version') {
            steps {
                sh '''
                python --version
                which python
                pip --version
                '''
            }
        }
        stage('Model Training') {
            steps {
                sh 'dvc repro --force'
            }
        }
        stage('Run Unit Tests') {
            steps {
                sh '''
                    python tests/conf_test.py
                    export PYTHONPATH=$WORKSPACE:$PYTHONPATH && pytest -v --disable-warnings --maxfail=1 --cov=mnist-dvc --cov-report=xml --cov-report=term
                '''
            }
        }
    }   
}
