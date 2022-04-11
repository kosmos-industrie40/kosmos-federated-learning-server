"""
This implementation providing all functionalities necessary
to manage the federated learning process by using Federated Averaging
and the bearing data.
This is the modified version of flower's FedAvg function:
https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

# from .aggregate import aggregate, weighted_loss_avg
# from .strategy import Strategy

import mlflow
import pandas as pd

from fl_models.DataSetType import DataSetType
from fl_models.CNNSpectraFeatures import CNNSpectraFeatures
from fl_models.util.metrics import rmse, correlation_coefficient
from fl_models.util.dataloader import read_feature_dfs_as_dict

from fl_server.util.helper_methods import pop_labels

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format
    loss, accuracy
move to
    loss, {"accuracy": accuracy}
instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type
    flwr.common.Weights (i.e., List[np.ndarray])
will be removed in a future update, move to
    flwr.common.Parameters
instead. Use
    parameters = flwr.common.weights_to_parameters(weights)
to easily transform `Weights` to `Parameters`.
"""


class FedAvgPlus(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        test_bearings: List[str],
        fraction_fit: float = 1.0,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        n_federated_training_rounds: int = 1,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """Federated Averaging strategy.
        Implementation based on https://arxiv.org/abs/1602.05629
        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__()
        self.test_bearings = test_bearings
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.n_federated_training_rounds: int = n_federated_training_rounds
        self.eval_fn = self.eval_fn_bearing
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.degradation_model: CNNSpectraFeatures = CNNSpectraFeatures(
            name="CNN_" + "global"
        )
        self.df_dicts: Dict[DataSetType, Dict[str, pd.DataFrame]] = None
        self.labels: Dict[DataSetType, Dict[str, pd.Series]] = None
        self.current_fed_rnd = 0

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters

    def load_data(self):
        """
        Loads the bearing data used for training in memory further to train the model
        :return:
        """
        self.df_dicts: Dict[DataSetType, Dict[str, pd.DataFrame]] = {}
        self.labels: Dict[DataSetType, Dict[str, pd.Series]] = {}
        for d_type in DataSetType:
            tmp_data = read_feature_dfs_as_dict(
                bearing_names=self.test_bearings,
                data_set_type=self.degradation_model.data_set_type,
            )
            tmp_labels = pop_labels(tmp_data)
            self.df_dicts[
                d_type
            ] = tmp_data  # {D_Type: {Bearing_Name: pd.DataFrame mit bearing observations}}
            self.labels[d_type] = tmp_labels

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)

        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            print(DEPRECATION_WARNING)
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics

    # MANUALLY ADDED FUNCTION
    def eval_fn_bearing(self, weights: Weights) -> Optional[Tuple[float, float]]:
        """Defining the model evaluation function."""

        if self.df_dicts is None:
            self.load_data()

        self.degradation_model.set_parameters(weights)
        d_type = self.degradation_model.data_set_type

        metrics = self.degradation_model.compute_metrics(
            df_dict=self.df_dicts.get(d_type),
            labels=self.labels.get(d_type),
            metrics_list=[rmse, correlation_coefficient],
        )

        denominator = len(metrics)
        metrics_names = ["RMSE", "PCC"]
        mean = dict.fromkeys(metrics_names, 0)
        for metric in metrics_names:
            metric_sum = 0
            for idx, (_, bearing_results) in enumerate(metrics.items()):
                metric_sum += bearing_results[metric]

                mlflow.log_metric(
                    metric + "_Client_" + str(idx),
                    bearing_results[metric],
                    self.current_fed_rnd,
                )

            mean[metric] = metric_sum / denominator

        mlflow.log_metric("RMSE", mean.get("RMSE"), self.current_fed_rnd)
        mlflow.log_metric("PCC", mean.get("PCC"), self.current_fed_rnd)

        self.current_fed_rnd += 1

        # return mean.get("RMSE"), mean.get("PCC") #deprecated ?
        return mean.get("RMSE"), {"accuracy": mean.get("PCC")}

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        return weights_to_parameters(aggregate(weights_results)), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}
