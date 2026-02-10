pipeline {
    agent any
    
    stages {
        stage("Install Dependencies") {
            steps {
                sh """
                python -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt

                which dvc
                dvc --version
                """
            }
        }
        stage('Model Training') {
            steps {
                sh """
                . venv/bin/activate
                dvc repro --force
                """
            }
        }
        stage('Run Unit Tests') {
            steps {
                sh '''
                    . venv/bin/activate
                    python tests/conf_test.py
                    export PYTHONPATH=$WORKSPACE:$PYTHONPATH && pytest -v --disable-warnings --maxfail=1 --cov=mnist-dvc --cov-report=xml --cov-report=term
                '''
            }
        }
        stage('Build Image') {
            steps {
                sh 'docker build -t deepakkumarkhatri/mnist-dvc-hub:latest .'
            }
        }
        stage('Push Image') {
            steps {
                withCredentials([usernamePassword(
                credentialsId: 'dockerhub-creds',
                usernameVariable: 'DOCKER_USERNAME',
                passwordVariable: 'DOCKER_PASSWORD'
                )]) {
                sh '''
                    echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
                    docker push deepakkumarkhatri/mnist-dvc-hub:latest
                '''
                }
            }
        }
    }   
}
