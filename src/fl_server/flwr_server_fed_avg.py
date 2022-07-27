"""
This implementation providing all functionalities necessary
to manage the federated learning process by using Federated Averaging
and the bearing data.
This is the modified version of flower's FedAvg function:
https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py
"""


from flwr.server.strategy import FedAvg
from fl_models.abstract.abstract_usecase import FederatedLearningUsecase


class FedAvgPlus(FedAvg):
    "Extended Flower FedAvg to have an fl usecase"

    def __init__(
        self,
        usecase: FederatedLearningUsecase,
        *args,
        n_federated_training_rounds=1,
        **kwargs
    ) -> None:
        super().__init__(eval_fn=usecase.eval_fn, *args, **kwargs)
        self.usecase = usecase
        self.n_federated_training_rounds = n_federated_training_rounds
