"""
This scripts starte a flower server using a customized flower training strategy
"""
from typing import List

import flwr as fl
import mlflow
from fl_server.flwr_server_fed_avg import FedAvgPlus


# Start Flower server for ten rounds of federated learning
# pylint: disable= too-many-arguments
def start_server(
    num_clients: int,
    test_bearings: List[str],
    num_fed_rounds: int,
    flwr_server_address: str = "localhost:8080",
    ml_flow_server_address: str = "./mlruns",
    ml_flow_run=None,
):
    """

    :param num_clients: Number of clients necessary to start training
    :param test_bearings: List of bearing names to evaluate global model on
    :param num_fed_rounds: Number of federaed training rounds
    :param flwr_server_address: The address this server is hosted at
    :param ml_flow_server_address: The mlflow tracking URI either a folder path or a network address
    :param ml_flow_run: mlflow run ID the client will save the logging to.
            All clients and the server should log to the same run
    :return:
    """

    # Check if ML-Flow server is up and running
    # connect to ML-Flow-Server

    mlflow.set_tracking_uri(ml_flow_server_address)
    if ml_flow_run is not None:
        mlflow.start_run(run_id=ml_flow_run, nested=True)

    mlflow.log_param("test_bearings", test_bearings)
    mlflow.log_param("number_clients", num_clients)
    mlflow.log_param("number_of_federated_trainingrounds", num_fed_rounds)

    # Create strategy
    strategy = FedAvgPlus(
        test_bearings,
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=num_clients,
        min_eval_clients=num_clients,
        min_available_clients=num_clients,
        n_federated_training_rounds=num_fed_rounds,
        # eval_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
    )

    stringlist = []
    # pylint: disable= unnecessary-lambda
    strategy.degradation_model.prediction_model.summary(
        print_fn=lambda x: stringlist.append(x)
    )
    mlflow.log_param("model_architecture", stringlist)
    fl.server.start_server(
        flwr_server_address,
        config={"num_rounds": strategy.n_federated_training_rounds},
        strategy=strategy,
    )


if __name__ == "__main__":
    default_test_bearings: List[str] = [
        "Bearing1_3",
        "Bearing1_4",
        "Bearing1_5",
        "Bearing1_6",
        "Bearing1_7",
        "Bearing2_3",
        "Bearing2_4",
        "Bearing2_5",
        "Bearing2_6",
        "Bearing2_7",
        "Bearing3_3",
    ]
    start_server(num_clients=3, test_bearings=default_test_bearings, num_fed_rounds=10)
