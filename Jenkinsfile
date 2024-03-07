pipeline {
    // use Docker as agent
    agent {
        docker {
            // Docker file attached below. Image pushed to dockerhub
            image 'efimovaleksey/mlops:stable'
            args '-u root:sudo '
        }
    }

    stages {
        stage('prepare_repo') {
            steps {
                sh "pwd"
                sh "python3 -m venv venv"
                sh ". venv/bin/activate"
                sh "pip3 install -rrequirements.txt"

                // Copy secret file to builddir
                withCredentials([file(credentialsId: 'kaggle_id', variable: 'kaggle_id')]) {
                    sh "cp \$kaggle_id $WORKSPACE"
                }
                
                sh "ls -la"
            }
        }
        
        stage('model_preprocession') {
             steps {
             	 sh "./pipeline.sh model_preprocession"
            }
        }
        
        stage('data_creation') {
             steps {
                 sh "./pipeline.sh data_creation"
            }
        }
        
        stage('model_preparation') {
             steps {
                 sh "./pipeline.sh model_preparation"
            }
        }
        
        stage('model_testing') {
             steps {
                 sh "./pipeline.sh model_testing"
            }
        }
    }
}
