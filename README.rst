=================================
KOSMoS Federated Learning Server
=================================


DESCRIPTION
===========
This repository implements a federated learning server. The server is part of the KOSMoS Federated Learning Framework 
which consists of two additional components: the `KOSMoS Federated Learning Client <https://github.com/kosmos-industrie40/kosmos-federated-learning-client>`_ and the `KOSMoS Federated Learning Resources <https://github.com/kosmos-industrie40/kosmos-federated-learning-resources>`_ project.
This project is able to run with any arbitrary data set but by default is executed with the bearing data set. For further information on design principals take a look at the `blogpost <https://www.inovex.de/de/blog/federated-learning-implementation-into-kosmos-part-3/>`_ describing the whole project.


USE CASE
========
The general goal is to collect machine data from machine operators at the KOSMoS Edge and then collaboratively train a model to predict the remaining useful lifetime. This Federated Bearing use case implements this approach with the following restrictions:

- The data used for training is not collected by the machine operator but the bearing data set is manually distributed to the collaborating clients
- The current project can be deployed with the docker container provided in this project but isn't deployed in the current KOSMoS project
- The connection between clients and host isn't encrypted as of now. This can be enabled in the Wrapper (websocket) and flower (grpc) implementation quite easily.

Open Points are:

- The optional but useful security features Differential Privacy and Secure Multiparty Computation are not implemented yet.

BUILD & RUN
===========

Docker
******

All containers run within the same network. In case the network has not been created yet run:

.. code-block::

    docker network create fl_network


To start the KOSMoS Federated Learning Server for a federated learning session docker containers can be used. First build the KOSMoS server and the mlflow server containers:

.. code-block::

    docker build --rm -t kosmos_fl_server:latest -f Dockerfiles/kosmos_fl_server.Dockerfile .

    docker build --rm -t mlflow:latest -f Dockerfiles/mlflow.Dockerfile - < Dockerfiles/mlflow.Dockerfile #- < Dockerfiles/mlflow.Dockerfile avoid that context is copied to container


After building both containers they can be executed using docker by the following commands:

.. code-block::

    docker run -d --network fl_network --name mlflow  mlflow:latest

    docker run -d --network fl_network --name kosmos_fl_server kosmos_fl_server:latest

During and after training you can connect to the mlflow ui by visiting  `http://localhost:5000 <http://localhost:5000>`_ or the remote server address set in the configuration.

Without Docker
**************

Further to execute the server locally the following steps must be taken. This project was
developed using python version 3.8. The behavior with other versions is undefined. There are known issues with tensorflow 2.5 and 2.6.

1. Install all necessary python packages (install in a virtual environment if necessary):

.. code-block::

    pip install -r requirements.txt

2. Install the server

.. code-block::

    python setup.py install

3. Furthermore, to access the logged training process visit the MLFlow UI server `http://localhost:5000 <http://localhost:5000>`_. MLFlow must be up and running:

.. code-block::

    mlflow ui

4. After starting the mlflow server you can run the KOSMoS Federated Learning Server with the following command:

.. code-block::

    cd src/fl_server/
    python kosmos_fl_server.py


Clients 
****************
The steps above start the server. The client needs to be started separately. For further information refer to the `client repository <https://github.com/kosmos-industrie40/kosmos-federated-learning-client>`_.

Troubleshooting:
****************

- Training not starting: Make sure that the number of clients of connecting to the server matches ``num_clients`` in :code:`config.yaml`.
- No Progress bar is shown when loading the bearing data: Add :code:`-tty` argument to :code:`docker run`


CONFIG.YAML FILE
================

The upbringing of a federated learning server with flower is based on the :code:`config.yaml` file featuring the following parameters:

.. list-table:: Configuration Details
   :widths: 25 25 25 50
   :header-rows: 1

   * - Name
     - Values
     - Default
     - Description
   * - num_clients
     - 1 or higher
     - 3
     - Number of participating clients must match the number of train bearing sets
   * - flwr_server_address
     - ip_address:port
     - [::1]:50052
     - Address and Port of the central flower server. Uses IPv6.
   * - socketio_address
     - ip_address:port
     - 0.0.0.0:6000
     - Address and Port of the central socketio server.
   * - mlflow_server_address
     - url:port or a local directory
     - http://localhost:5000
     - Address to the MLFlow server
   * - mlflow_experiment_name
     - string
     - "Default"
     - The MLFlow experiment the federated training process will be logged to
   * - tags
     - List of one or more tag:value pairs
     - experiment_name: "default_config"
     - These tags will be associated with the MLRun and are important for filtering and comparing multiple federated learning runs
   * - n_federated_train_epoch
     - int >=1
     - 5
     - The number of federated learning iterations with all clients
   * - test_bearing
     - List of bearing names
     - ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3",]
     - The names of the bearings used for testing at the central flower server

Note that the bearings available for training and testing are chosen distinctively from the list of all available bearings.
Because of the nature of federated learning, a bearing should be used exclusively as test or as
client training data.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
