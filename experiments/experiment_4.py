import itertools

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


from src.MAO.covering import coverage_degress
from plots import plots, plot_evo
from metrics import metrics_base, metrics_MAO, find_best_k


def main():
    """
    Experiment 4: General case
    Study the average behavior of the indices as k increases. Compare with the true labels.

    """
    print("Hello from clusterqualitymad!")


    # Generate combinations for user-specific requirements
    k_values_custom = [2, 4, 8, 16, 32] #2
    p_values_custom = [2, 5, 10]
    n_values_custom = [128, 256, 512, 1024]
    d_intervals_custom = [(0.1, 0), (0.2, 1), (0.3, 2), (0.4, 3), (0.5, 4)]

    all_results = []
    for k, p, n, (sl, su) in itertools.product(k_values_custom, p_values_custom, n_values_custom, d_intervals_custom):
        scenario_name = f"P{p}-K{k}-N{n}-dt{sl:.2f}-S{su}"
        print(scenario_name)
        blob_file = f"data/blobs/blobs-{scenario_name}.npy"
        blobs = np.load(blob_file)
        scaler = StandardScaler()  # FIXME: Here or before clustering the data?
        scaler.fit(blobs[:-1])
        scaler.transform(blobs[:-1])
        X = blobs[:,:-1]
        y = blobs[:,-1].astype(int)
        results = []
        all_labels = [y]
        possible_k = range(2,50) # range(2,max(int(1.5*k),6))
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

        # Ejemplo aplicado a toda tu matriz (fila por fila)
        resultados_k = []
        for i in range(results.shape[1]):
            method = header[i]
            k_optimo = find_best_k(results[:, i], method)
            resultados_k.append(k_optimo)

        all_results.append([p,k,n,sl,su,*resultados_k])
    all_results = np.array(all_results)
    np.savetxt("output/experiment_4_results.csv", all_results, fmt="%s")
    print(header)



if __name__ == "__main__":
    main()
