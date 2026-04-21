import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import kmeans_plusplus

from fuzzy_cmeans import FuzzyCMeans
import skfuzzy as fuzz

from plots import plots, plot_evo
from metrics import metrics_base, metrics_MAO, find_best_k, curvature_ratios
from src.MAO.covering import coverage_degress

def kmean_pp(X, n_clusters, random_state=42):
    initial_centers, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=42)
    # From centers to U
    dists = np.linalg.norm(X[:, np.newaxis] - initial_centers, axis=2)
    dists = np.fmax(dists, 1e-10)
    u_init = 1.0 / (dists ** 2)
    u_init = u_init / u_init.sum(axis=1)[:, np.newaxis]
    return u_init.T

def main():
    """
    Experiment 3: membership and coverage degrees
    This experiment is the same as experiment 1/2 but using Fuzzy C-Means instead of K-Means.
    And, then the matrix U is the one obtained from Fuzzy C-Means instead for XB and from coverage_degrees for MEOWAs.
    Study one dataset and the behavior of the indices as k increases. Compare with the true labels.

    """
    print("Experiment 3: membership and coverage degrees for one example.")
    N = 512
    actual_clusters = 4
    dt = 0.3 # 0.3
    S = 2  # 2
    blob_file = f"data/blobs/blobs-P2-K{actual_clusters}-N{N}-dt{dt}0-S{S}-test.npy"
    blobs = np.load(blob_file)
    X_raw = blobs[:, :-1]
    y = blobs[:, -1].astype(int)
    X = StandardScaler().fit_transform(X_raw)
    results_base = []
    results_MAO_m = []
    results_MAO_cv = []
    all_labels = [y]
    possible_k = range(2,15) # actual_clusters+3)
    for n_clusters in possible_k: # Davies Bouldin and Silhouette need at least 2 clusters
        u_init = kmean_pp(X, n_clusters, random_state=42)
        centroids, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, n_clusters, 2, error=1e-10, maxiter=10000, init=u_init, seed=42)

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

        # Actualizaci�n de centroides
        labels = np.argmax(U_fuzz, axis=1)
        # centroids = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                centroids[i] = np.mean(points_in_cluster, axis=0)
            else:
                centroids[i] = centroids[i]



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

    # study k=1 for MEOWAs and curvature method
    k1_centroid = np.mean(X, axis=0).reshape(1, -1)
    U_k1_m = np.ones((X.shape[0], 1))
    result_MAO_k1_m = metrics_MAO(U_k1_m)
    U_k1_cv = coverage_degress(X, k1_centroid)
    result_MAO_k1_cv = metrics_MAO(U_k1_cv)


    resultados_k = []
    header = []
    for name, results in results_dic.items():
        metrics = np.array(results["values"]).T
        for data, method in zip(metrics,results["headers"]):
            if "_m" in name:
                data = np.insert(data, 0, result_MAO_k1_m[method])
            if "_cv" in name:
                data = np.insert(data, 0, result_MAO_k1_cv[method])
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

    epsilon = 1e-10
    for name, results in results_dic.items():
        if name == "base": # Only for the base metrics, not for the MAO ones, because they are not supposed to employ curvature
            continue
        headers = results["headers"]
        metrics = np.array(results["values"]).T
        ratios_all = []
        for data, method in zip(metrics,headers):
            ratios = curvature_ratios(data)
            ratios_all.append(ratios)
        ratios_all = np.array(ratios_all).T
        possible_k = list(range(2, 1+len(data))) # It starts in k=2
        plot_evo(ratios_all, headers, possible_k, title=f"Experiment_3_RatiosN_{name}", nexp=3)


if __name__ == "__main__":
    main()
