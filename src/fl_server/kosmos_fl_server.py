"""
This script starts a server that implements the usecase configured in config.yaml. The server
triggers the federated training if enough clients participate.
"""
import os
from multiprocessing import Process
from threading import Lock
from warnings import warn

import eventlet
import mlflow
import socketio
from dynaconf import Dynaconf
from fl_server.flwr_server import start_server

# import threading
# from eventlet.green import threading
# eventlet.monkey_patch()

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
CONFIG = Dynaconf(settings_files=[CONFIG_FILE])

client_list = []
client_list_lock = Lock()
TRAIN_CLIENT_LIST = []
train_clients_list_lock = Lock()
sio = socketio.Server(logger=CONFIG.DEBUG, engineio_logger=CONFIG.DEBUG)
app = socketio.WSGIApp(
    sio  # , static_files={"/": {"content_type": "text/html", "filename": "index.html"}}
)
app = socketio.WSGIApp(sio, app)


@sio.event
# pylint: disable= unused-argument
def connect(sid, data):
    """
    This default function is called if a connection from a client has been established
    """
    with client_list_lock:
        client_list.append(sid)


@sio.event
def message(sid, data):
    """
    This is the default message handler if no handler has been specified for a event.
    It should not be used in the production code but is useful for debugging.
    :param data:
    """
    print(f"message from {sid} with data {data}")


@sio.event
def client_criteria(sid, data):
    """
    Client response to it's criteria check. If it's successful the client can be added to the
    client training pool. As soon as a specified number of suitable clients are in the pool
    the federated training will be started.
    :param sid: client session id
    :param data: result of client data criteria check
    """
    # pylint: disable= global-statement
    global TRAIN_CLIENT_LIST
    # check if criteria has been met
    criteria_are_met = data.get("criteria_are_met")

    if criteria_are_met is None:
        # mising information from client
        warn("Missing key 'criteria_are_met' in transmitted data. Ignoring client.")
        return

    if criteria_are_met is False:
        return

    with train_clients_list_lock:
        TRAIN_CLIENT_LIST.append(sid)
        if len(TRAIN_CLIENT_LIST) == CONFIG.get("num_clients"):
            print("Enough clients available start training now")
            train_on_all_clients(CONFIG)
            TRAIN_CLIENT_LIST = []


@sio.event
def disconnect(sid):
    """
    The default event handler called if the connection is closed.
    NOTE: It won't be called if the connection is interrupted unintendedly
    """
    print(f"Client {sid} disconnected from server")


def train_on_all_clients(flwr_config: Dynaconf):
    """
    This function triggers the training event on all clients. Therefore it sends the
    central model further to be trained on the private data of the clients.

    :param flwr_config: Loaded configuration. Has the following mandatory keys:
        * mlflow_server_address
        * mlflow_experiment_name
        * num_clients
        * n_federated_train_epoch
        * flwr_server_address
        * usecase
            * name
            * params
    """
    server_address = flwr_config["mlflow_server_address"]
    mlflow.set_tracking_uri(server_address)

    mlflow.end_run()

    # exp_id = self._mlflow_create_experiment_if_not_exist(self.mlflow_experiment_name)
    mlflow.set_experiment(flwr_config["mlflow_experiment_name"])

    mlflow.start_run()
    mlflow_run_id = mlflow.active_run().info.run_id
    if flwr_config.get("tags") is not None:
        print(flwr_config["tags"])
        for key, value in flwr_config["tags"].items():
            mlflow.set_tag(key, value)

    broadcast_config = flwr_config.as_dict()["USECASE"].get("broadcast", {})

    joined_config = {**flwr_config.as_dict()["USECASE"]["params"], **broadcast_config}

    # print("============= Starting federated learning server =============")

    flower_server_process = Process(
        target=start_server,
        args=(
            flwr_config["num_clients"],
            flwr_config["usecase"]["name"],
            flwr_config["n_federated_train_epoch"],
            flwr_config["flwr_server_address"],
            server_address,
            mlflow_run_id,
        ),
        kwargs=joined_config,
    )

    flower_server_process.start()

    for idx, sid in enumerate(TRAIN_CLIENT_LIST):
        sio.emit(
            "start_train",
            {
                "client_id": idx,
                "mlflow_experiment_name": flwr_config["mlflow_experiment_name"],
                "mlflow_server_address": server_address,
                "mlflow_run_id": mlflow_run_id,
                "flwr_server_address": flwr_config["flwr_server_address"],
                "usecase_name": flwr_config["usecase"]["name"],
                "usecase_params": broadcast_config,
            },
            room=sid,
        )
        print(f"sent training request to client {idx} with sid {sid}")
    eventlet.sleep(0)

    flower_server_process.join()


if __name__ == "__main__":
    # Starts the usecase server which waits for a specified number of participating clients
    # and then runs the federated learning process using flower. The clients can be forced to check
    # their data for certain conditions further to attend in the training process (number of
    # training data, length of sequences, ...).
    # The server will be started at localhost:6000 if there is no other address and port specified
    # in the python arguments or the environment variables. The environment variables will be used
    # before the parsed argument which will be used before default settings. The environment
    # variables must be named according to the argument parser destination.

    PATH = None

    address = CONFIG.socketio_address.split(":")
    print(f"Connecting server to address: {CONFIG.socketio_address}")
    print(f"mlflow address: {CONFIG.mlflow_server_address}")
    print(f"flwr server: {CONFIG.flwr_server_address}")
    eventlet.wsgi.server(eventlet.listen((address[-2], int(address[-1]))), app)
