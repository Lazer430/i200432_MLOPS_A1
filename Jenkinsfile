pipeline {
  agent any
  options {
    buildDiscarder(logRotator(numToKeepStr: '2'))
  }
  environment {
    DOCKERHUB_CREDENTIALS = credentials('1234')
  }
  stages {
    stage('Build') {
      steps {
        sh 'docker build -t ahsanbaloch/mlops_a1:latest .'
      }
    }
    stage('Login') {
      steps {
        sh 'echo $DOCKERHUB_CREDENTIALS_PSW | docker login -u $DOCKERHUB_CREDENTIALS_USR --password-stdin'
      }
    }
    stage('Push') {
      steps {
        sh 'docker push ahsanbaloch/mlops_a1:latest'
      }
    }
  }
  post {
    always {
      sh 'docker logout'
      emailext body: 'Build succeeded and pushed to docker hub', replyTo: 'i200432@nu.edu.pk,i200474@nu.edu.pk', subject: 'Build succeeded and pushed to docker hub', to: 'i200432@nu.edu.pk,i200474@gmail.com,ahsanrasheed0474@gmail.com'
    }
  }
}