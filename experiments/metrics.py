import numpy as np
from scipy import stats

from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    calinski_harabasz_score,
)
from src.MAO import MAO
from src.MAO.covering import (
    Xie_Beni_Index,
    Partition_Coefficient,
    Partition_Entropy,
    Xie_Beni_Hard,
)

def curvature_method(matrix_values, threshold=0.05):
    """Curvature method

    """

    # 1. Calculamos las ganancias (primeras diferencias)
    gains = np.abs(np.diff(matrix_values))

    # --- REGLA ESPECIAL PARA K=2 ---
    # Normalizamos respecto al valor inicial (o al rango) para ver si el primer salto es relevante
    # FIXME: Check this out
    first_gain_relative = gains[0] / (np.max(matrix_values) + 1e-10)

    if first_gain_relative < threshold:
        return 2  # La curva es plana desde el principio

    # 2. Calculamos las ratios de desaceleración (Tail Ratios)
    # Evitamos dividir por cero añadiendo un epsilon muy pequeño
    ratios = gains[:-1] / (gains[1:] + 1e-10)

    # 3. El mejor k es el índice del máximo ratio + offset
    # (El offset depende de dónde empezara tu k. Si la col 0 es k=2...)
    best_idx = np.argmax(ratios)

    # Si matrix_values empezaba en k=2:
    # gains[0] es k3-k2, gains[1] es k4-k3...
    # ratios[0] evalúa el codo en k=3
    return best_idx + 3


def find_best_k(matrix_values, method, threshold=0.05):
    """
    matrix_values: array de una fila con los valores del índice para k=2,3,4...
    """
    if method in ["DB","PE","XB", "XBH"]:
        # Min value
        return np.argmin(matrix_values) + 2  # +2 porque k empieza en 2
    elif method in ["PC","SS", "CH"]:
        # max value
        return np.argmax(matrix_values) + 2  # +2 porque k empieza en 2

    return curvature_method(matrix_values, threshold)


def metrics_base(U, V, X, y, w):
    pc = Partition_Coefficient(U, m=2)
    pe = Partition_Entropy(U)
    xb = Xie_Beni_Index(X, V, U)
    labels = np.argmax(U, axis=1)
    DB = davies_bouldin_score(U, labels)
    CH = calinski_harabasz_score(X, labels)
    SS = silhouette_score(U, labels)
    xbh = Xie_Beni_Hard(X, V, labels)
    # ARI = adjusted_rand_score(labels, y)
    # NMI = normalized_mutual_info_score(labels, y)
    # ACC = accuracy_score(labels, y)
    results = {"PC": pc, "PE": pe, "XB": xb, "XBH": xbh, "DB": DB, "SS": SS, "CH": CH,
               # "ARI": ARI, "NMI": NMI, "ACC":ACC,
               }
    return results

def metrics_MAO(U):
    MM = MAO.MAO_Max_Max(U)
    # Mm = MAO.MAO_Max_Min(U)
    mM = MAO.MAO_Min_Max(U)
    MeM = MAO.MAO_Mean_Max(U)
    results = {
               "MAO_Min_Max": mM, "MAO_Max_Max": MM, "MAO_Mean_Max": MeM,
               # "MAO_Max_Min": Mm
               }
    oreness = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.55,0.65,0.75,0.85,0.95]
    N = len(U)
    for o in oreness:
        w_file = f"weights/W_{N}_{o}.npy"
        w = np.load(w_file)
        MEOWA = MAO.MAO_MEOWA(U, w, A=np.max)
        results[f"MEOWA_{o}"] = MEOWA

    return results
