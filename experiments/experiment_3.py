import numpy as np
from sklearn.preprocessing import StandardScaler

from fuzzy_cmeans import FuzzyCMeans
import skfuzzy as fuzz

from plots import plots, plot_evo
from metrics import metrics_base, metrics_MAO, find_best_k
from src.MAO.covering import coverage_degress


def main():
    """
    Experiment 3: membership and coverage degrees
    This experiment is the same as experiment 1/2 but using Fuzzy C-Means instead of K-Means.
    And, then the matrix U is the one obtained from Fuzzy C-Means instead for XB adn from coverage_degrees for MEOWAs.
    Study one dataset and the behavior of the indices as k increases. Compare with the true labels.

    """
    print("Experiment 3: membership and coverage degrees for one example.")
    N = 1024
    actual_clusters = 4
    dt = 0.5 # 0.3
    S = 4  # 2
    blob_file = f"data/blobs/blobs-P2-K{actual_clusters}-N{N}-dt{dt}0-S{S}-test.npy"
    blobs = np.load(blob_file)
    scaler = StandardScaler()  # FIXME: Here or before clustering the data?
    scaler.fit(blobs[:-1])
    scaler.transform(blobs[:-1])
    X = blobs[:,:-1]
    y = blobs[:,-1].astype(int)
    results_base = []
    results_MAO_m = []
    results_MAO_cv = []
    all_labels = [y]
    possible_k = range(2,10) # actual_clusters+3)
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
        U_fuzz = u.T  # 2d array, (N, c) -> Final fuzzy c - partitioned matrix. Transpuesta de u para que sea n_samples x n_clusters
        # Other lib
        # fcm = FuzzyCMeans(n_clusters=n_clusters, m=2, random_state=42).fit(X)
        # centroids = fcm.cluster_centers_
        # U = fcm.U_
        labels = np.argmax(U_fuzz, axis=1)
        all_labels.append(labels)
        result_base = metrics_base(U_fuzz, centroids, X, y, "w")
        result_MAO_m = metrics_MAO(U_fuzz)
        U_cd = coverage_degress(X, centroids) # cd: coverage degree
        result_MAO_cv = metrics_MAO(U_cd)
        # Not in fuzzy-CM
        # SSE = fcm.inertia_
        # result["SSE"] = SSE
        results_base.append(list(result_base.values()))
        results_MAO_m.append(list(result_MAO_m.values()))
        results_MAO_cv.append(list(result_MAO_cv.values()))

    results_dic= {"base": {"values": results_base, "headers": list(result_base.keys())},
           "MAO_m": {"values": results_MAO_m, "headers": list(result_MAO_m.keys())},
           "MAO_cv": {"values": results_MAO_cv, "headers": list(result_MAO_cv.keys())}}

    for name, results in results_dic.items():
        plots(X, all_labels, results["values"], results["headers"], name, nexp=3)

    resultados_k = []
    header = []
    for name, results in results_dic.items():
        metrics = np.array(results["values"]).T
        for data, method in zip(metrics,results["headers"]):
            k_optimo = find_best_k(data, method)
            resultados_k.append(k_optimo)
            header.append(f"{name}_{method}")

    for method, k in zip(header, resultados_k):
        print(f"Using {method}, the best k is: {k} ({actual_clusters=})")

    epsilon = 1e-10
    for name, results in results_dic.items():
        metrics = np.array(results["values"])
        header = results["headers"]
        gains_current = metrics[:-1, :]  # filas 0, 1, 2
        gains_next = metrics[1:, :]  # filas 1, 2, 3
        ratios = gains_current / (gains_next + epsilon)
        best_k_indices = np.argmax(ratios, axis=0)
        k_reales = best_k_indices + 2 # It starts in k=2

        possible_k = list(range(2, 1+len(metrics))) # It starts in k=2
        plot_evo(ratios, header, possible_k, title=f"Experiment_3_Ratios_{name}", nexp=3)


if __name__ == "__main__":
    main()
