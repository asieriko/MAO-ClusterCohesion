import numpy as np
from sklearn.preprocessing import StandardScaler

from fuzzy_cmeans import FuzzyCMeans
import skfuzzy as fuzz

from plots import plots, plot_evo
from metrics import metrics_base, metrics_MAO, find_best_k


def main():
    """
    Experiment 2: Particular case  Fuzzy C-Means.
    This experiment is the same as experiment 1 but using Fuzzy C-Means instead of K-Means.
    And, thus the matrix U is the one obtained from Fuzzy C-Means instead of the one obtained from coverage_degress.
    Study one dataset and the behavior of the indices as k increases. Compare with the true labels.

    """
    print("Experiment 2: Fuzzy C-Means")
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
        centroids, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None, seed=42)
        # centroids: 2d array, size(c, S) -> Cluster         centers.Data for each center along each feature provided for every cluster(of the `c` requested clusters).
        # u: 2d array, (c, N) -> Final fuzzy c - partitioned matrix.
        # u0: 2d array, (c, N) -> Initial guess at fuzzy c - partitioned matrix(either provided init or random guess used if init was not provided).
        # d: 2d array, (c, N) -> Final Euclidian distance matrix.
        # jm: 1d array, length P -> Objective function history.
        # p: int -> Number of iterations run.
        # fpc: float -> Final Fuzzy partition coefficient.
        U = u.T  # 2d array, (N, c) -> Final fuzzy c - partitioned matrix. Transpuesta de u para que sea n_samples x n_clusters
        # Other lib
        # fcm = FuzzyCMeans(n_clusters=n_clusters, m=2, random_state=42).fit(X)
        # centroids = fcm.cluster_centers_
        # U = fcm.U_
        labels = np.argmax(U, axis=1)
        all_labels.append(labels)
        result = metrics_base(U, centroids, X, y, "w")
        result = result | metrics_MAO(U)
        # Not in fuzzy-CM
        # SSE = fcm.inertia_
        # result["SSE"] = SSE
        results.append(list(result.values()))

    results = np.array(results)
    header = list(result.keys())
    plots(X, all_labels, results, header, "Cluster Quality Indices FCM", nexp=2)
    plot_evo(results[:,-8:],header[-8:], possible_k,title="MAO FCM", nexp=2)
    plot_evo(results[:, :-8], header[:-8], possible_k,title="Non-MAO FCM", nexp=2)

    # Ejemplo aplicado a toda tu matriz (fila por fila)
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

    plot_evo(ratios, header, possible_k, title="Ratios", nexp=2)

if __name__ == "__main__":
    main()
