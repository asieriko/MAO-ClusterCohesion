import itertools
from pathlib import Path


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import kmeans_plusplus
import skfuzzy as fuzz

from src.MAO.covering import coverage_degress
from metrics import metrics_base, metrics_MAO, find_best_k
from experiment_4_summary import summary_exp4


def methods_analysis(df: pd.DataFrame):
    df_bool = pd.concat(
        [
            df.iloc[:, :4],  # primeras 4 columnas (su fuera)
            df.iloc[:, 5:].eq(df["k"], axis=0)  # comparación vectorizada
        ],
        axis=1
    )

    param_cols = ["p","k","n","sl"]  # su fuera
    method_cols = [c for c in df.columns if c not in param_cols]
    # Mejor método por fila
    df["best_method"] = df[method_cols].idxmax(axis=1)

    # Peor método por fila
    df["worst_method"] = df[method_cols].idxmin(axis=1)
    df["best_method"].value_counts()
    df["worst_method"].value_counts()
    df.groupby("k")["best_method"].value_counts(normalize=True)
    pd.crosstab(df["k"], df["best_method"])


def kmean_pp(X, n_clusters, random_state=42):
    initial_centers, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=42)
    # From centers to U
    dists = np.linalg.norm(X[:, np.newaxis] - initial_centers, axis=2)
    dists = np.fmax(dists, 1e-10)
    u_init = 1.0 / (dists ** 2)
    u_init = u_init / u_init.sum(axis=1)[:, np.newaxis]
    return u_init.T

def main(lim=50):
    """
    Experiment 4: General case
    Study the average behavior of the indices as k increases. Compare with the true labels.

    This experiment is the same as experiment 3 but for multiple examples.
    The matrix U is the one obtained from Fuzzy C-Means instead for XB ann from coverage_degrees for MEOWAs.
    Study mutiple dataset and the behavior of the indices as k increases. Compare with the true labels.

    """
    print("Experiment 4: membership and coverage degrees for multiple examples.")
    # Generate combinations for user-specific requirements
    k_values_custom = [4, 8, 16, 32] #2
    possible_k_dict = {2:10, 4:15, 8:25, 16:35, 32:35}
    #k_values_custom = [v for v in k_values_custom if v <= lim]
    p_values_custom = [2, 5, 10, 15, 20]
    n_values_custom = [128, 256, 512, 1024]
    d_intervals_custom = [(0.1, 0), (0.2, 1), (0.25, 0), (0.3, 2), (0.35, 9), (0.4, 3), (0.5, 4)]

    cache_dir = Path("temp_cache")
    cache_path_dir = Path(cache_dir).resolve()
    cache_path_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for k, p, n, (sl, su) in itertools.product(k_values_custom, p_values_custom, n_values_custom, d_intervals_custom):
        scenario_name = f"P{p}-K{k}-N{n}-dt{sl:.2f}-S{su}"
        print(scenario_name)
        blob_file = f"data/blobs/blobs-{scenario_name}.npy"
        blobs = np.load(blob_file)
        X_raw = blobs[:, :-1]
        y = blobs[:, -1].astype(int)
        X = StandardScaler().fit_transform(X_raw)
        results = []
        all_labels = [y]

        possible_k = range(2, possible_k_dict[k]) # = range(2,50) # range(2, max(k * 2, 10))
        for n_clusters in possible_k: # Davies Bouldin and Silhouette need at least 2 clusters

            file_name = f"cache_{scenario_name}-NC{n_clusters}.npz"
            file_path = cache_path_dir / file_name

            if file_path.exists():
                # Cargar datos desde disco
                with np.load(file_path) as data:
                    centroids = data['centroids']
                    U_fuzz = data['U_fuzz']
                    U_cd = data['U_cd']
            else:
                # Ejecutar clustering fuzzy c-means
                u_init = kmean_pp(X, n_clusters, random_state=42)
                centroids, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                    X.T, n_clusters, 2, error=1e-10, maxiter=10000, init=u_init, seed=42)
                U_fuzz = u.T

                # Actualizaci�n de centroides
                labels = np.argmax(U_fuzz, axis=1)
                # centroids = np.zeros((n_clusters, X.shape[1]))
                for i in range(n_clusters):
                    points_in_cluster = X[labels == i]
                    if len(points_in_cluster) > 0:
                        centroids[i] = np.mean(points_in_cluster, axis=0)
                    else:
                        centroids[i] = centroids[i]

                U_cd = coverage_degress(X, centroids)

                # Guardar matrices para evitar recalcular en el futuro
                np.savez_compressed(file_path,
                                    centroids=centroids,
                                    U_fuzz=U_fuzz,
                                    U_cd=U_cd)


            # centroids, u, u0, d, jm, it, fpc = fuzz.cluster.cmeans(
            #     X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None, seed=42)
            # U_fuzz = u.T

            labels = np.argmax(U_fuzz, axis=1)
            all_labels.append(labels)

            result = metrics_base(U_fuzz, centroids, X, y, "w")
            result_MAO = metrics_MAO(U_fuzz)
            result = result | {k:v for k,v in zip([f"{x}_m" for x in result_MAO.keys()], result_MAO.values())}
            # U_cd = coverage_degress(X, centroids)  # cd: coverage degree
            result_MAO = metrics_MAO(U_cd)
            result = result | {k:v for k,v in zip([f"{x}_cv" for x in result_MAO.keys()], result_MAO.values())}

            results.append(list(result.values()))

        results = np.array(results)
        header = list(result.keys())


        # study k=1 for MEOWAs and curvature method
        k1_centroid = np.mean(X, axis=0).reshape(1, -1)
        U_k1_m = np.ones((X.shape[0], 1))
        result_MAO_k1_m = metrics_MAO(U_k1_m)
        U_k1_cv = coverage_degress(X, k1_centroid)
        result_MAO_k1_cv = metrics_MAO(U_k1_cv)

        resultados_k = []
        for i in range(results.shape[1]):
            method = header[i]
            data = results[:, i]
            if "_m" in method:
                data = np.insert(data, 0, result_MAO_k1_m[method[:-2]])
            if "_cv" in method:
                data = np.insert(data, 0, result_MAO_k1_cv[method[:-3]])
            k_optimo = find_best_k(data, method)
            resultados_k.append(k_optimo)

        all_results.append([p,k,n,sl,su,*resultados_k])
    all_results = np.array(all_results)
    headers = ["p","k","n","sl","su"] + header
    df = pd.DataFrame(all_results, columns=headers)
    summary_exp4(df)
    df_bool = pd.concat(
        [
            df.iloc[:, :4],  # primeras 4 columnas (su, no)
            df.iloc[:, 5:].eq(df["k"], axis=0)  # comparación vectorizada
        ],
        axis=1
    )
    str_headers = (";".join(headers))
    np.savetxt(f"output/experiment_4_results_{lim}.csv", all_results, delimiter=";", header=str_headers, fmt="%s",
               comments="")
    print(header)




if __name__ == "__main__":
    main(50)
    # main()

