#!/usr/bin/env bash
#
source /etc/profile
pwdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z "$DOCKER_NAME" ]]; then
        DOCKER_NAME="login-model-longwindow-docker"
fi

if [[ -z "$DOCKER_TAG" ]]; then
        DOCKER_TAG="login-model-longwindow"
fi

docker run -d -t -p 10006:8080 --name $DOCKER_NAME $DOCKER_TAG serve
