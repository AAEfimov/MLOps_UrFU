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
              
		sh "dvc remote modify myremote --local gdrive_user_credentials_file gdrive.json" 
		withCredentials([file(credentialsId: 'gdrive', variable: 'gdrive')]) {
		    sh "cp \$gdrive $WORKSPACE"
                }
 
                sh "ls -la"
            }
        }
       
	stage('code_testing') {
	     steps {
		sh "echo Test"
	     }
	}

	stage('dvc_data_get') {
	     steps {
		sh 'rm stages/model.pkl'
		sh "dvc pull"
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

	stage('data_testing') {
	     steps {
		sh "./pipeline.sh test_data"
	    }
	}
    }
}
