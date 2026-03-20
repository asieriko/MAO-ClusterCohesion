import numpy as np
from numpy.ma.core import append
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from scipy import stats

from fuzzy_cmeans import FuzzyCMeans

import matplotlib.pyplot as plt

from src.MAO.covering import Xie_Beni_Index, Partition_Coefficient, Partition_Entropy, coverage_degress
from src.MAO import MAO

def test_weights():
    oreness = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]
    samples = [64,128,256,512,1024]
    for o in oreness:
        for s in samples:
            w_file = f"weights/W_{s}_{o}.npy"
            w = np.load(w_file)
            print(w.shape)


def metrics(U, V, X, w):
    xb = Xie_Beni_Index(X, V, U)
    pc = Partition_Coefficient(U, m=2)
    pe = Partition_Entropy(U)
    MM = MAO.MAO_Max_Max(U)
    Mm = MAO.MAO_Max_Min(U)
    mM = MAO.MAO_Min_Max(U)
    labels = np.argmax(U, axis=1)
    DB = davies_bouldin_score(U, labels)
    SS = silhouette_score(U, labels)
    w_file = f"weights/W_64_0.45.npy"
    w = np.load(w_file)
    MEOWA = MAO.MAO_MEOWA(U, w, A=np.max)
    print(f"XB: {xb}\nPC: {pc}\nPE: {pe}"
          f"\nMAO_Max_Max: {MM}\nMAO_Max_Min: {Mm}\nMAO_Min_Max: {mM}\nMAO_MEOWA: {MEOWA}\n"
          f" DB: {DB}, SS: {SS}")


def metrics_non_fuzz(U, y, w):
    pc = Partition_Coefficient(U, m=2)
    pe = Partition_Entropy(U)
    labels = np.argmax(U, axis=1)
    DB = davies_bouldin_score(U, labels)
    SS = silhouette_score(U, labels)
    ARI = adjusted_rand_score(labels, y)
    NMI = normalized_mutual_info_score(labels, y)
    ACC = accuracy_score(labels, y)
    MM = MAO.MAO_Max_Max(U)
    Mm = MAO.MAO_Max_Min(U)
    mM = MAO.MAO_Min_Max(U)
    results = {"PC": pc, "PE": pe, "DB": DB, "SS": SS, "ARI": ARI, "NMI": NMI, "ACC":ACC, "MAO_Max_Max": MM, "MAO_Max_Min": Mm, "MAO_Min_Max": mM}
    oreness = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]
    N = len(U)
    for o in oreness:
        w_file = f"weights/W_{N}_{o}.npy"
        w = np.load(w_file)
        MEOWA = MAO.MAO_MEOWA(U, w, A=np.max)
        results[f"MEOWA_{o}"] = MEOWA
    print(f"PC: {pc}\nPE: {pe}\nDB: {DB}\nSS: {SS}\nMAO_Max_Max: {MM}\nMAO_Max_Min: {Mm}\nMAO_Min_Max: {mM}\nMAO_MEOWA: {MEOWA}")
    return results

def plot_metrics(data, headers):
    plt.figure(figsize=(10, 6))
    num_cols = data.shape[1]
    ks = range(2, 2 + data.shape[0])  # Assuming k starts from 2
    for i in range(num_cols):
        plt.plot(ks, data[:, i], label=headers[i])
    plt.xlabel("k")
    plt.ylabel("Value")
    plt.title("Metrics vs. k")
    plt.ylim(0, 2)  # Adjust y-axis limits if necessary
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    plt.savefig("metrics_vs_k.png")


def correlation_heatmap(data, headers):
    fig, ax = plt.subplots(figsize=(15, 15))
    # ax = plt.gca()
    im = ax.imshow(data)
    ax.figure.colorbar(im, ax=ax, shrink=0.74)
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # fig.colorbar(im, cax=cbar_ax)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(len(headers)), labels=headers,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(headers)), labels=headers)

    # Loop over data dimensions and create text annotations.
    for i in range(len(headers)):
        for j in range(len(headers)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title("Spearman correlation")
    fig.tight_layout()
    plt.show()
    plt.savefig("correlation_heatmap.png")

def main():
    print("Hello from clusterqualitymad!")

    N = 1024
    actual_clusters = 4
    blob_file = f"data/blobs/blobs-P2-K{actual_clusters}-N{N}-dt0.30-S2.npy"
    blobs = np.load(blob_file)
    scaler = StandardScaler()  # FIXME: Here or before clustering the data?
    scaler.fit(blobs[:-1])
    scaler.transform(blobs[:-1])
    X = blobs[:,:-1]
    y = blobs[:,-1].astype(int)
    meoas = []
    pcs = []
    results = []
    for n_clusters in range(2,11): # Davies Bouldin and Silhouette need at least 2 clusters
        # no tiene sentido determinar así el nº de clusters, porque al hacer máximos siempre será mejor cuantos más clusters
        # si hay un cluster por cada punto, el resultado es 1
        # lo que mide es lo bien asignados que están los puntos. o algo así.
        fcm = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        centroids = fcm.cluster_centers_
        U = coverage_degress(X, centroids) # n_samples x n_clusters
        result = metrics_non_fuzz(U, y, "w")
        results.append(list(result.values()))
        pc = Partition_Coefficient(U, m=2)
        w_file = f"weights/W_{N}_0.45.npy"
        w = np.load(w_file)
        MEOWA = MAO.MAO_MEOWA(U, w, A=np.max)
        meoas.append(MEOWA)
        pcs.append(pc)

    results = np.array(results)
    header = list(result.keys())
    # plot_metrics(results,header)
    print(header)
    print(results)
    res = stats.spearmanr(results)[0]
    print(f"{res=}")
    correlation_heatmap(res, header)
    print(meoas,pcs)
    print(f"{actual_clusters}: MEOWAS {np.argmax(meoas) + 1}, PCs {np.argmax(pcs)+1}")

    # fcm = FuzzyCMeans(n_clusters=3, m=2, random_state=42)
    # fcm.fit(X)
    # # U = [n_samples x n_clusters]
    # U = fcm.U_
    # # V = [n_clusters x n_features]
    # V = fcm.cluster_centers_
    # # U,V = WFCM(X, y, c=2, m=2, it_max=100)#, random_state=42)
    # metrics(U, V, X, "w")



if __name__ == "__main__":
    main()
