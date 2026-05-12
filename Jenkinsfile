pipeline {
    agent any

    parameters {
        string(name: 'EC2_HOST', defaultValue: '127.0.0.1', description: 'DNS/IP instanta AWS EC2; 127.0.0.1 cand Jenkins ruleaza pe aceeasi instanta')
        string(name: 'EC2_USER', defaultValue: 'ubuntu', description: 'User SSH EC2, de obicei ubuntu pentru Ubuntu AMI')
        string(name: 'SSH_CREDENTIALS_ID', defaultValue: 'aws-ec2-ssh-key', description: 'ID credential SSH din Jenkins')
        string(name: 'APP_PORT', defaultValue: '8000', description: 'Port public pe EC2')
    }

    environment {
        APP_NAME = 'licenta-plant-api'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        IMAGE_TAR = 'plant-diagnosis-api.tar.gz'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Static Check') {
            steps {
                sh 'python3 -B -m py_compile app/api.py src/predict_image.py src/multicrop_voting.py src/symbolic_rules.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'DOCKER_BUILDKIT=0 docker build -t ${APP_NAME}:${IMAGE_TAG} .'
                sh 'docker save ${APP_NAME}:${IMAGE_TAG} | gzip > ${IMAGE_TAR}'
            }
        }

        stage('Deploy To AWS EC2') {
            steps {
                sshagent(credentials: ["${params.SSH_CREDENTIALS_ID}"]) {
                    sh '''
                        set -e
                        scp -o StrictHostKeyChecking=no "${IMAGE_TAR}" "${EC2_USER}@${EC2_HOST}:/tmp/${IMAGE_TAR}"
                        scp -o StrictHostKeyChecking=no deploy/ec2_deploy.sh "${EC2_USER}@${EC2_HOST}:/tmp/ec2_deploy.sh"
                        ssh -o StrictHostKeyChecking=no "${EC2_USER}@${EC2_HOST}" \
                          "chmod +x /tmp/ec2_deploy.sh && APP_NAME=${APP_NAME} IMAGE_NAME=${APP_NAME}:${IMAGE_TAG} IMAGE_TAR=/tmp/${IMAGE_TAR} APP_PORT=${APP_PORT} /tmp/ec2_deploy.sh"
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'rm -f ${IMAGE_TAR}'
        }
    }
}
