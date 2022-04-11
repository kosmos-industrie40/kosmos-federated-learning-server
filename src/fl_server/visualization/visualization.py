"""
File contains visualization methods.
"""
import os
from typing import Dict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from tensorflow.keras.callbacks import History

from fl_models.util.metrics import standard_deviation
from fl_server.util.constants import LOG_PICTURES_PATH


def plot_trainings_history(trainings_history: History, error_type: str = "MAE"):
    """
    Plots the training history of a keras model.
    :param trainings_history: keras history object
    :param error_type: string that includes loss name
    :return: Void, shows plot
    """
    plt.plot(trainings_history.history["loss"], label=error_type + " (training data)")
    # plt.plot(trainings_history.history['val_loss'], label=error_type + ' (validation data)')

    plt.title(error_type + " for RUL prediction")
    plt.ylabel(error_type + " value")
    plt.xlabel("No. epoch")
    plt.legend(loc="upper left")
    plt.show()


def plot_rul_comparisons(
    bearing_data: Dict[str, pd.DataFrame],
    label_data: Dict[str, pd.Series],
    prediction_model,
    experiment_name: str = None,
    model_name: str = None,
):
    """
    Plot the real RUL in comparison to the RUL predicted by a Keras Model of multiple data frames.
    :param bearing_data: list of feature_dfs which RULs are to be predicted
    :param prediction_model: model used for prediction
    :return: Void, plots Facet grid which plots predicted and real RUL for each data frame
    """
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.figure(figsize=(12, 9))
    num_bearings: int = len(bearing_data)
    sqr: int = isqrt(num_bearings)

    prediction_dict = prediction_model.predict(bearing_data)
    count = 1
    for key in prediction_dict.keys():
        predictions = prediction_dict[key]
        rul = label_data[key]
        # Smooth predictions
        predictions = savgol_filter(predictions, 9, 3)
        # predictions = linear_rectification_technique(pd.Series(predictions))
        plt.subplot(sqr, sqr, count)
        sns.lineplot(data=rul)
        sns.lineplot(x=rul.index, y=predictions, size=0.1)
        plt.xlabel("Observation")
        plt.ylabel("RUL in Seconds")
        plt.legend([], [], frameon=False)
        plt.title(key.replace("_", " "))
        count += 1
    plt.tight_layout()
    if model_name is None or experiment_name is None:
        plt.show()
    else:
        path_out = (
            Path(LOG_PICTURES_PATH).joinpath("rul_comparison").joinpath(experiment_name)
        )
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        path_out = path_out.joinpath(model_name + ".png")
        plt.savefig(path_out, dpi=300)
        plt.clf()

    return path_out


# Helper
# pylint: disable= invalid-name
def isqrt(number):
    """
    Computes an approximate integer to the sqrt
    """
    x = number
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + number // x) // 2
    if x * x == number:
        return x
    return x + 1


def plot_frequency_heatmap(zxx, f, t):
    """
    Plot frequency heatmap
    """
    plt.pcolormesh(t, f, np.abs(zxx), shading="gouraud")
    plt.title("Spectrum Magnitude")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.tight_layout()
    plt.show()


def plot_metric_bar_overview(
    metric_data: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str,
    experiment_name: str = None,
):
    """
    Plots a metric bar
    """
    # set width of bar
    bar_width = 10
    space_between_groups = bar_width * 2
    groups = list(
        metric_data.get(list(metric_data.keys())[0]).keys()
    )  # get bearing keys

    # Set amount of members per group
    group_members = metric_data.keys()
    x_bar_left = np.array(
        [
            (bar_width * len(group_members) + space_between_groups) * i
            for i in range(len(groups))
        ]
    )

    offset = -len(group_members) / 2
    for member in group_members:
        y_values = [
            metric_dict.get(metric_key)
            for bearing, metric_dict in metric_data.get(member).items()
        ]
        plt.bar(
            x_bar_left + offset * bar_width,
            y_values,
            width=bar_width,
            label=member.replace("_", " "),
            edgecolor="black",
        )
        offset += 1

    plt.ylabel(metric_key)
    plt.xlabel("Bearings")
    plt.xticks(x_bar_left, [group[7:].replace("_", " ") for group in groups])
    plt.legend()

    if experiment_name is None:
        plt.savefig("test.png")
        plt.show()
    else:
        path_out = (
            Path(LOG_PICTURES_PATH)
            .joinpath("metrics_bar_plot")
            .joinpath(experiment_name)
        )
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        path_out = path_out.joinpath(metric_key + ".png")
        plt.savefig(path_out, dpi=300)
        plt.clf()


# pylint: disable= too-many-locals
def plot_aggregated_metrics(
    metric_data: Dict[str, Dict[str, Dict[str, float]]], experiment_name: str = None
):
    """
    Plots a metric bar of the aggregated metrics
    """
    bar_width = 0.5
    # plt.tight_layout()
    models = list(metric_data.keys())
    bearings = list(metric_data.get(models[0]).keys())
    metrics = list(metric_data.get(models[0]).get(bearings[0]).keys())
    x = np.arange(len(models))
    subplot_count = 1
    for metric_key in metrics:
        plt.subplot(1, len(metrics), subplot_count)
        count = 0
        for model in models:
            model_metrics = metric_data.get(model)
            metric_values_list = []
            for bearing in model_metrics.keys():
                metric_values_list += [model_metrics.get(bearing).get(metric_key)]
            std_dev = standard_deviation(metric_values_list)
            plt.bar(
                x[count],
                height=sum(metric_values_list) / len(metric_values_list),
                width=bar_width,
                yerr=std_dev,
            )
            count += 1

        plt.ylabel(metric_key)
        plt.xlabel("Models")
        plt.xticks(x, [model.replace("_", " ") for model in models], fontsize=12)
        subplot_count += 1
    if experiment_name is None:
        plt.show()
    else:
        path_out = Path(LOG_PICTURES_PATH).joinpath("aggregated_metrics_bar_plot")
        if not os.path.exists(path_out):
            Path(path_out).mkdir(parents=True, exist_ok=True)
        path_out = path_out.joinpath(experiment_name)
        plt.savefig(path_out, dpi=300)
        plt.clf()
