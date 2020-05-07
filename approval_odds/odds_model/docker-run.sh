#!/usr/bin/env bash
#
source /etc/profile
pwdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z "$DOCKER_NAME" ]]; then
        DOCKER_NAME="ctr-model-loggedin-ccsearchtool-docker"
fi

if [[ -z "$DOCKER_TAG" ]]; then
        DOCKER_TAG="ctr-model-loggedin-ccsearchtool"
fi

docker run -d -t -p 10004:8080 --name $DOCKER_NAME $DOCKER_TAG serve
