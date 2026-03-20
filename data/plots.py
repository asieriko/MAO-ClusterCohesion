import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def plot_cluster(X, y, title="Cluster Plot"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=40)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.savefig(f"data/figures/{title}.png")

k_values_custom = [2, 4, 8, 16, 32]
p_values_custom = [2]#, 5, 10]
n_values_custom = [128, 256, 512, 1024]
d_intervals_custom = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

for actual_clusters in k_values_custom:
    for sample in n_values_custom:
        for i,d in enumerate(d_intervals_custom):
            blob_file = f"data/blobs/blobs-P2-K{actual_clusters}-N{sample}-dt{d:.2f}-S{i}.npy"
            blobs = np.load(blob_file)
            scaler = StandardScaler()  # FIXME: Here or before clustering the data?
            scaler.fit(blobs[:-1])
            scaler.transform(blobs[:-1])
            X = blobs[:,:-1]
            y = blobs[:,-1].astype(int)
            plot_cluster(X,y, blob_file[10:-4])

