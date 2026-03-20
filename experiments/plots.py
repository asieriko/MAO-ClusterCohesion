import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def plots(X, y, result, names, title="Cluster Quality Indices", nexp=1):
    """
    Creates a plot with 2 rows and K columns, where K is the number of clusters.
    the first row contains scatter plots of the data colored by the true labels and the predicted labels for each K.
    (the first plot is the true labels, and the next K-1 plots are the predicted labels for K=...)
    the second row contains tables with the values of the indices for each K.

    Parameters
    ----------
    X : data points
    y : actual data labels
    result : array of shape (K-1, n_indices) with the values of the indices for each K (starting from K=2)
    names: names of the indices

    Returns
    -------
    Saves a file
    """

    K = len(y)

    fig = plt.figure(figsize=(18, 10))

    # 2 filas, 4 columnas
    gs = GridSpec(
        2, K,
        width_ratios=K * [2],   # columna 0 más estrecha
        height_ratios=[3, 1],               # gráficos arriba, tabla abajo
        figure=fig
    )

    # --- Fila superior: gráficos en columnas 1, 2, 3 ---

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(X[:, 0], X[:, 1], c=y[0], cmap="tab10")
    ax.set_title(f"True")

    for i in range(1, K):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(X[:, 0], X[:, 1], c=y[i], cmap="tab10")
        ax.set_title(f"K = {i+1}")

    # --- Fila inferior: tabla dividida en 4 columnas ---

    # Columna 0: títulos
    ax_t0 = fig.add_subplot(gs[1, 0])
    ax_t0.axis("off")
    table0 = ax_t0.table(
        cellText=[[r] for r in names],
        colLabels=[""],
        loc="center",
        cellLoc="left"
    )
    table0.auto_set_font_size(False)
    table0.set_fontsize(11)
    table0.scale(1, 1.4)

    # Columna 1: valores 1
    for i in range(1, K):
        ax_t1 = fig.add_subplot(gs[1, i])
        ax_t1.axis("off")
        table1 = ax_t1.table(
            cellText=[[f"{v:.2f}"] for v in result[i-1]],
            colLabels=[f"K = {i+1}"],
            loc="center",
            cellLoc="center"
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(11)
        table1.scale(1, 1.4)



    plt.tight_layout()
    # plt.show()
    plt.savefig(f"output/experiment_{nexp}_{title.replace(" ","_")}.png")

def plot_evo(data, header, ks,title="Metrics", nexp=1):
    """
    Plots a line plot for each column of the data

    Parameters
    ----------
    data: array of shape (n_k, n_indices) with the values of the indices for each K (starting from K=2)
    header: names of the indices, for the legend
    ks: values of k corresponding to the rows of data
    title: title of the plot

    Returns
    -------

    """
    n_points, n_series = data.shape
    x = np.arange(n_points)  # or your own x values

    plt.figure(figsize=(10, 6))

    for i in range(n_series):
        plt.plot(x, data[:, i], label=header[i])

    plt.xlabel("K")
    plt.ylabel("Ratio")
    plt.title("Ratios of metrics for consecutive K values")
    plt.xticks(range(len(ks)),labels=ks)
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"output/experiment_{nexp}_{title}.png")
