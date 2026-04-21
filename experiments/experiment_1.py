import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from fuzzy_cmeans import FuzzyCMeans

from src.MAO.covering import coverage_degress
from plots import plots, plot_evo
from metrics import metrics_base, metrics_MAO, find_best_k


def main():
    """
    Experiment 1: Particular case K-Means.
    This experiment uses K-Means instead of FCM.
    And, thus the matrix U is the one obtained from coverage_degress instead of the one obtained from Fuzzy C-Means.
    Study one dataset and the behavior of the indices as k increases. Compare with the true labels.

    """
    print("Experiment 1: K-Means")
    N = 256
    actual_clusters = 4
    dt = 0.4
    S = 3
    blob_file = f"data/blobs/blobs-P2-K{actual_clusters}-N{N}-dt{dt}0-S{S}.npy"
    blobs = np.load(blob_file)
    X_raw = blobs[:, :-1]
    y = blobs[:, -1].astype(int)
    X = StandardScaler().fit_transform(X_raw)
    results = []
    all_labels = [y]
    possible_k = range(2,actual_clusters+3)
    for n_clusters in possible_k: # Davies Bouldin and Silhouette need at least 2 clusters
        fcm = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        centroids = fcm.cluster_centers_
        U = coverage_degress(X, centroids) # n_samples x n_clusters
        labels = np.argmax(U, axis=1)
        all_labels.append(labels)
        result = metrics_base(U, centroids, X, y, "w")
        result = result | metrics_MAO(U)
        SSE = fcm.inertia_
        result["SSE"] = SSE
        results.append(list(result.values()))

    results = np.array(results)
    header = list(result.keys())
    plots(X, all_labels, results, header, nexp=1)
    plot_evo(results[:,-8:],header[-8:], possible_k,title="MAO", nexp=1)
    plot_evo(results[:, :-8], header[:-8], possible_k,title="Non-MAO", nexp=1)

    # Ejemplo aplicado a toda la matriz (fila por fila)
    resultados_k = []
    for i in range(results.shape[1]):
        method = header[i]
        k_optimo = find_best_k(results[:, i], method)
        resultados_k.append(k_optimo)

    for method, k in zip(header, resultados_k):
        print(f"Using {method}, the best k is: {k} ({actual_clusters=})")

    epsilon = 1e-10
    gains_current = results[:-1, :]  # filas 0, 1, 2
    gains_next = results[1:, :]  # filas 1, 2, 3
    ratios = gains_current / (gains_next + epsilon)
    best_k_indices = np.argmax(ratios, axis=0)
    k_reales = best_k_indices + 2 # It starts in k=2

    plot_evo(ratios, header, possible_k, title="Ratios", nexp=1)

if __name__ == "__main__":
    main()
