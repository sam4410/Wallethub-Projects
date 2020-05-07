# Docker to host the Approval Odds model

The docker is hosting a gradient boosting model developed using scikit-learn library. The docker is created using a frame-work provided by Sagemaker.

SageMaker supports two execution modes: _training_ where the algorithm uses input data to train a new model and _serving_ where the algorithm accepts HTTP requests and uses the previously trained model to do an inference (also called "scoring", "prediction", or "transformation"). Model is trained offline and the docker is used only for scoring. 

In order to build a production grade inference server into the container, use the following stack to make the implementer's job simple:

1. __[nginx][nginx]__ is a light-weight layer that handles the incoming HTTP requests and manages the I/O in and out of the container efficiently.
2. __[gunicorn][gunicorn]__ is a WSGI pre-forking worker server that runs multiple copies of your application and load balances between them.
3. __[flask][flask]__ is a simple web framework used in the inference app that you write. It lets you respond to call on the `/ping` and `/invocations` endpoints without having to write much code.

## The Structure of the Code

The components are as follows:

* __Dockerfile__: The _Dockerfile_ describes how the image is built and what it contains. It is a recipe for your container and gives you tremendous flexibility to construct almost any execution environment you can imagine. Here. we use the Dockerfile to describe a pretty standard python science stack and the simple scripts that we're going to add to it. See the [Dockerfile reference][dockerfile] for what's possible here.

* __im-approval_odds__: The directory that contains the application to run in the container. See the next session for details about each of the files.

* __Scirpt-BuildDocker-Sagemaker__: This is the script that is used to create the docker on SageMaker. This script creates the docker and pushes the image to ECR Repo. The current image is pushed to repository [434418615032.dkr.ecr.us-east-2.amazonaws.com/approval-odds-allcards-tmp]

### The application run inside the container

When IM starts a container, it will invoke the container with an argument of either __train__ or __serve__. We have set this container only for the __serve__ program.

* __train__: The main program for training the model. When you build your own algorithm, you'll edit this to include your training code. (not included in the current files)
* __serve__: The wrapper that starts the inference server. In most cases, you can use this file as-is.
* __wsgi.py__: The start up shell for the individual server workers. This only needs to be changed if you changed where predictor.py is located or is named.
* __predictor.py__: The algorithm-specific inference server. This is the file that you modify with your own algorithm's code.
* __nginx.conf__: The configuration for the nginx master server that manages the multiple workers.

### External files accessed by the container
The docker is accessing two external files from the S3 bucket __efmldevelopment__ located in __/data/creditcards/__. 

* __card_variables.csv__: This csv have details about card variables 

The latest files are updated in the docker using the route - updateccdata. (command to run > curl -X GET <localhost:port>/updateccdata) 

## Environment variables

When you create an inference server, you can control some of Gunicorn's options via environment variables. These
can be supplied as part of the CreateModel API call.

    Parameter                Environment Variable              Default Value
    ---------                --------------------              -------------
    number of workers        MODEL_SERVER_WORKERS              the number of CPU cores
    timeout                  MODEL_SERVER_TIMEOUT              60 seconds


[skl]: http://scikit-learn.org "scikit-learn Home Page"
[dockerfile]: https://docs.docker.com/engine/reference/builder/ "The official Dockerfile reference guide"
[ecr]: https://aws.amazon.com/ecr/ "ECR Home Page"
[nginx]: http://nginx.org/
[gunicorn]: http://gunicorn.org/
[flask]: http://flask.pocoo.org/

