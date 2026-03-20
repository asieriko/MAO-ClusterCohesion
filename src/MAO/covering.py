import numpy as np
from scipy.spatial.distance import cdist

def coverage_degress(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Computes the coverage degrees of data points to cluster centroids.

     .. math::
            U = [u_{ij}] where

            u_{ij} = e^{-\frac{r}{s} d(x_i, c_j)}

            r = 2 \cdot ln(10)

            s = 5 \cdot \sqrt(D)

            D = dimensionality of the data

    Parameters:
    x (np.ndarray): The dataset used for clustering.
    centroids (np.ndarray): The centroids of the clusters.

    Returns:
    np.ndarray: The coverage degrees matrix.
    """
    # u_ij se calcula  a partir de los datos estandarizados(StandardScaler) con la expresión(1) del adjunto,
    # usando s = 5 * sqrt(D), con D la dimensión de los datos(número de variables / componentes) y r = 2ln10.
    N = x.shape[0]  # Number of data points
    K = centroids.shape[0]  # Number of clusters
    r = 2 * np.log(10)
    D = x.shape[1]  # Dimensionality of the data
    s = 5 * np.sqrt(D)

    U = np.zeros((N, K))

    for i in range(N):
        for j in range(K):
            dist_ij = np.linalg.norm(x[i] - centroids[j])
            U[i, j] = np.exp(-r/s * dist_ij)

    return U

def Xie_Beni_Index(data: np.ndarray, centroids: np.ndarray, U: np.ndarray) -> float:
    """
    Computes the Xie-Beni Index for evaluating clustering quality.

    The Xie-Beni Index is defined as:
    XB = (\sum_i^N \sum_j^K u_{ij}^2 \cdot d(o_i, c_j)^2 / (N * d_min^2)
    where:
    - u_{ij} is the membership degree of data point i to cluster j,
    - d(o_i, c_j) is the distance between data point i and centroid j
    - O_i is the i-th data point,
    - c_j is the j-th cluster centroid,
    - N is the number of data points,
    - K is the number of clusters,
    - d_min is the minimum distance between cluster centroids.

    A lower XB value indicates better clustering quality.

    Parameters:
    data (np.ndarray): The dataset used for clustering.
    centroids (np.ndarray): The centroids of the clusters.
    U (np.ndarray): The membership matrix.

    Returns:
    float: The computed Xie-Beni Index.
    """
    N = data.shape[0]  # Number of data points
    K = centroids.shape[0]  # Number of clusters

    # Compute pairwise distances between centroids
    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(centroids)

    # Set diagonal to infinity to ignore zero distances
    np.fill_diagonal(distances, np.inf)

    # Find the minimum distance between any two centroids
    d_min = np.min(distances)

    # Compute the Xie-Beni Index
    #T = (\sum_i^N \sum_j^K u_{ij}^2 \cdot d(o_i, c_j)
    compactness = 0.0
    for i in range(N):
        for j in range(K):
            dist_ij = np.linalg.norm(data[i] - centroids[j])
            compactness += (U[i, j] ** 2) * dist_ij**2

    xb_index = compactness / (N * (d_min**2))

    return xb_index


def Xie_Beni_Hard(data:np.ndarray, centroids:np.ndarray, labels: np.ndarray) -> float:
    """
    Xie-Beni para clustering nítido (Hard Clustering).
    labels: array de tamaño N con el índice del cluster asignado (0, 1, ..., K-1)
    """
    N = data.shape[0]

    # --- 1. Numerador: Compacidad Total (SSE) ---
    compactness = 0.0
    for j in range(len(centroids)):
        # Filtramos solo los puntos que pertenecen al cluster j
        cluster_points = data[labels == j]
        if len(cluster_points) > 0:
            # Sumamos las distancias al cuadrado de esos puntos a su centroide
            sq_dists = np.sum(np.linalg.norm(cluster_points - centroids[j], axis=1) ** 2)
            compactness += sq_dists

    # --- 2. Denominador: Separación Mínima ---
    centroid_sq_dists = cdist(centroids, centroids, metric='sqeuclidean')
    np.fill_diagonal(centroid_sq_dists, np.inf)
    min_sep_sq = np.min(centroid_sq_dists)

    return compactness / (N * min_sep_sq)

def Partition_Coefficient(U: np.ndarray, m: float=2) -> float:
    """
    Computes the Partition Coefficient for evaluating clustering quality.
    Journal of Mathematical Biology; Article. Numerical taxonomy with fuzzy sets.
    Published: May 1974. Volume 1, pages 57–71, (1974) J. C. Bezdek
    https://doi.org/10.1007/BF02339490

    The PC is defined as:
    XB = 1/n (\sum_i^k \sum_j^n u_{ij}^m)
    where:
    - u_{ij} is the membership degree of data point i to cluster j,
    - k is the number of clusters,
    - n is the number of data points
    - m is the fuzziness parameter. (usually 2)

    When the value of PC is larger, it indicates that the degree of overlap between the clusters is smaller
    and the clusters are better separated

    Parameters:
    U (np.ndarray): The membership matrix.
    m (float): The fuzziness parameter. Default is 2.

    Returns:
    float: The computed PC Index.
    """
    n = U.shape[0]  # Number of data points
    m = 2  # Fuzziness parameter

    pc_index = np.sum(U ** m) / n

    return pc_index

def Partition_Entropy(U: np.ndarray) -> float:
    """
    Computes the Partition Entropy for evaluating clustering quality.
    Bezdek, James C.. “Cluster Validity with Fuzzy Sets.” (1973).
    https://doi.org/10.1080/01969727308546047

    The PC is defined as:
    XB = 1/n (\sum_i^k \sum_j^n u_{ij} * log_a(u_{ij}))
    where:
    - u_{ij} is the membership degree of data point i to cluster j,
    - k is the number of clusters,
    - n is the number of data points
    - a (log_a)

    when the value of PE is larger, it indicates that the degree of fuzziness of
    the clusters is higher and the clusters are more overlapping

    Parameters:
    U (np.ndarray): The membership matrix.

    Returns:
    float: The computed PE Index.
    """
    n = U.shape[0]  # Number of data points

    pe_index = np.sum(U * np.log(U)) / n

    return pe_index