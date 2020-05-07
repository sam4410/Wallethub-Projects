#!/usr/bin/env bash
#
source /etc/profile
pwdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
algorithm_name="login-model-longwindow"
docker_name="${algorithm_name}-docker"

mode="$1"
if [[ -z "$mode" ]]; then
	mode="build,run,ssh"
fi	

if [[ "$mode" == *"build"* ]]; then
	docker build . -t $algorithm_name
fi

if [[ "$mode" == *"run"* ]]; then
	docker stop $docker_name 
        docker system prune -f
        docker run -d -t -p 10007:8080 --name $docker_name $algorithm_name serve	
fi

if [[ "$mode" == *"ssh"* ]]; then
	docker exec -it $docker_name /bin/bash
fi
