pipeline {
    agent {
        docker {
            reuseNode false
            image 'ubuntu:latest'
        }
    }
    //triggers{
    //    cron('H H 1 1-12 *')
    }
    environment {
        BUILDSTARTDATE = sh(script: "echo `date +%Y%m%d`", returnStdout: true).trim()
        S3PROJECTDIR = '' // no trailing slash

        // Distribution ID for the AWS CloudFront for this bucket
        // used solely for invalidations
        AWS_CLOUDFRONT_DISTRIBUTION_ID = 'EUVSWXZQBXCFP'
    }
    options {
        timestamps()
        disableConcurrentBuilds()
    }
    stages {
        // Very first: pause for a minute to give a chance to
        // cancel and clean the workspace before use.
        stage('Ready and clean') {
            steps {
                // Give us a minute to cancel if we want.
                sleep time: 30, unit: 'SECONDS'
            }
        }

        stage('Initialize') {
            steps {
                // print some info
                dir('./working') {
                    sh 'env > env.txt'
                    sh 'echo $BRANCH_NAME > branch.txt'
                    sh 'echo "$BRANCH_NAME"'
                    sh 'cat env.txt'
                    sh 'cat branch.txt'
                    sh "echo $BUILDSTARTDATE"
                    sh "python --version"
                    sh "id"
                    sh "whoami" // this should be jenkinsuser
                    // if the above fails, then the docker host didn't start the docker
                    // container as a user that this image knows about. This will
                    // likely cause lots of problems (like trying to write to $HOME
                    // directory that doesn't exist, etc), so we should fail here and
                    // have the user fix this

                }
            }
        }

        stage('Setup') {
            steps {
                dir('./working') {
                    sh 'python -m pip install oaklib'
                // install s3cmd too
                }
            }
        }

        stage('Run similarity ') {
            // TODO: verify the version of PHENIO we are using and note that...somewhere
            steps {
                dir('./working') {
                    sh 'runoak -i sqlite:obo:hp descendants -p i HP:0000118 > HPO_terms.txt'
                    sh 'runoak -i sqlite:obo:mp descendants -p i MP:0000001 > MP_terms.txt'
                    sh 'runoak -i semsimian:sqlite:obo:phenio similarity -p i --set1-file HPO_terms.txt --set2-file MP_terms.txt -O csv -o HP_vs_MP_semsimian.tsv'
                }
            }
        }

        stage('Upload result') {
            // Store similarity results at s3://kg-hub-public-data/monarch/
            steps {
                dir('./gitrepo') {
                    script {

                        // make sure we aren't going to clobber existing data
                            withCredentials([
					            file(credentialsId: 's3cmd_kg_hub_push_configuration', variable: 'S3CMD_CFG'),
					            file(credentialsId: 'aws_kg_hub_push_json', variable: 'AWS_JSON'),
					            string(credentialsId: 'aws_kg_hub_access_key', variable: 'AWS_ACCESS_KEY_ID'),
					            string(credentialsId: 'aws_kg_hub_secret_key', variable: 'AWS_SECRET_ACCESS_KEY')]) {
                                                              

                                // upload to remote
                                sh 's3cmd -c $S3CMD_CFG put -pr --acl-public --cf-invalidate HP_vs_MP_semsimian.tsv s3://kg-hub-public-data/monarch/'

                                // Should now appear at:
                                // https://kg-hub.berkeleybop.io/monarch/
                            }

                        }
                    }
                }
            }
        }

    }

    post {
        always {
            echo 'In always'
            echo 'Cleaning workspace...'
            cleanWs()
        }
        success {
            echo 'I succeeded!'
        }
        unstable {
            echo 'I am unstable :/'
        }
        failure {
            echo 'I failed :('
        }
        changed {
            echo 'Things were different before...'
        }
    }
}