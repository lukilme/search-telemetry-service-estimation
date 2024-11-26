import pandas as pd
from typing import Tuple
import numpy as np
import matplotlib as plot
from scipy import stats
import mypy
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_results(feature_importances, feature_names):

    feature_importances_df = pd.DataFrame(
        {"Features": feature_names, "Importância": feature_importances}
    ).sort_values(by="Importância", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Importância", y="Features", data=feature_importances_df)
    plt.title("Importância das Features")
    plt.xlabel("Importância")
    plt.ylabel("Features")
    plt.show()


def plot_predictions_comparison(
    start_graph, end_graph, labels_list, predictions_list, model_name
):
    datasets_info = [
        (
            "Sinusoid",
            labels_list[0][start_graph:end_graph],
            predictions_list[0][start_graph:end_graph],
        ),
        (
            "Flashcrowd",
            labels_list[1][start_graph:end_graph],
            predictions_list[1][start_graph:end_graph],
        ),
        (
            "Mix",
            labels_list[2][start_graph:end_graph],
            predictions_list[2][start_graph:end_graph],
        ),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16), constrained_layout=True)
    t = np.arange(start_graph, end_graph)

    ax1.plot(t, datasets_info[0][1], label="Sinusoid Labels", color="blue")
    ax1.plot(t, datasets_info[0][2], label="Sinusoid Predictions", color="red")
    ax1.set_title("Sinusoid: Labels vs Predictions")
    ax1.set_xlabel("Índice")
    ax1.set_ylabel("Valor")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(t, datasets_info[1][1], label="Flashcrowd Labels", color="blue")
    ax2.plot(t, datasets_info[1][2], label="Flashcrowd Predictions", color="red")
    ax2.set_title("Flashcrowd: Labels vs Predictions")
    ax2.set_xlabel("Índice")
    ax2.set_ylabel("Valor")
    ax2.grid(True)
    ax2.legend()

    ax3.plot(t, datasets_info[2][1], label="Mix Labels", color="blue")
    ax3.plot(t, datasets_info[2][2], label="Mix Predictions", color="red")
    ax3.set_title("Mix: Labels vs Predictions")
    ax3.set_xlabel("Índice")
    ax3.set_ylabel("Valor")
    ax3.grid(True)
    ax3.legend()

    plt.show()


def visualize_results(feature_importances, feature_names):
    feature_importances_df = pd.DataFrame(
        {"Features": feature_names, "Importância": feature_importances}
    ).sort_values(by="Importância", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Importância", y="Features", data=feature_importances_df)
    plt.title("Importância das Features")
    plt.xlabel("Importância")
    plt.ylabel("Features")
    plt.show()
