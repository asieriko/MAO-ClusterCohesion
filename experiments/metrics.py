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

def curvature_ratios_range(matrix_values):
    """
    Calcula los ratios de curvatura y la importancia del primer salto.
    """
    # 1. Ganancias absolutas (diferencias entre k y k+1)
    gains = np.abs(np.diff(matrix_values))

    # 2. Importancia relativa del primer salto (k=2 -> k=3)
    # Comparamos la ganancia con el rango total para saber si "añadir clústeres"
    # aporta algo significativo desde el principio.
    total_range = np.max(matrix_values) - np.min(matrix_values) + 1e-10
    first_gain_relative = gains[0] / total_range
    # first_gain_relative = gains[0] / (np.max(matrix_values) + 1e-10)

    # 3. Ratios de desaceleración (Tail Ratios)
    # Añadimos un epsilon para evitar divisiones por cero
    # ratios[0] comparará salto(2->3) / salto(3->4) -> evalúa el codo en k=3
    ratios = gains[:-1] / (gains[1:] + 1e-10)

    return ratios, first_gain_relative, gains


def curvature_method_k2(matrix_values, threshold=0.01):  # Sugerencia: bajar a 0.01 para MEOWA
    """
    Determina el mejor k usando los ratios, gestionando k=2 y el ruido de cola.
    """
    ratios, first_gain_relative, gains = curvature_ratios_range(matrix_values)

    # --- LÓGICA PARA K=2 ---
    # Si lo que ganamos al pasar de 2 a 3 clústeres es despreciable (menor al umbral),
    # significa que con 2 ya hemos capturado la estructura principal.
    if first_gain_relative < threshold:
        return 2

    # --- FILTRO DE RUIDO DE COLA (Tail Noise) ---
    # En rangos largos (k=2 a 50), las ganancias al final son tan pequeñas (0.00001)
    # que sus ratios pueden dar picos falsos por precisión numérica.
    # Ignoramos ratios donde la ganancia sea menor al 1% de la ganancia máxima.
    max_gain = np.max(gains)
    valid_ratios_mask = gains[1:] > (max_gain * 0.02)

    # Si ninguna ganancia tras el primer salto es significativa, nos quedamos con 2
    if not np.any(valid_ratios_mask):
        return 2

    # 3. El mejor k es el índice del máximo ratio + offset
    # (El offset depende de dónde empezara tu k. Si la col 0 es k=2...)
    # best_idx = np.argmax(ratios) # NOTE: old version

    # Aplicamos la máscara: ponemos a 0 los ratios que vienen de fluctuaciones ínfimas
    filtered_ratios = ratios.copy()
    filtered_ratios[~valid_ratios_mask] = 0

    # Buscamos el máximo ratio entre los válidos
    best_idx = np.argmax(filtered_ratios)

    # best_idx = 0 -> ratios[0] (salto 2-3 / 3-4) -> k=3
    return best_idx + 3


def curvature_method_k1(matrix_values, threshold=0.01):  # Sugerencia: bajar a 0.01 para MEOWA
    """
    Determina el mejor k usando los ratios, gestionando k=2 y el ruido de cola.
    """
    ratios, first_gain_relative, gains = curvature_ratios_range(matrix_values)


    # --- FILTRO DE RUIDO DE COLA (Tail Noise) ---
    # En rangos largos (k=2 a 50), las ganancias al final son tan pequeñas (0.00001)
    # que sus ratios pueden dar picos falsos por precisión numérica.
    # Ignoramos ratios donde la ganancia sea menor al 1% de la ganancia máxima.
    max_gain = np.max(gains)
    valid_ratios_mask = gains[1:] > (max_gain * 0.02)

    # 3. El mejor k es el índice del máximo ratio + offset
    # (El offset depende de dónde empezara tu k. Si la col 0 es k=2...)
    # best_idx = np.argmax(ratios) # NOTE: old version

    # Aplicamos la máscara: ponemos a 0 los ratios que vienen de fluctuaciones ínfimas
    filtered_ratios = ratios.copy()
    filtered_ratios[~valid_ratios_mask] = 0

    # Buscamos el máximo ratio entre los válidos
    best_idx = np.argmax(filtered_ratios)

    # best_idx = 0 -> ratios[0] (salto 2-3 / 3-4) -> k=3
    return best_idx + 2

def curvature_ratios(matrix_values):
    """
    Calcula los ratios de curvatura y la importancia del primer salto.
    """
    # 1. Ganancias absolutas (diferencias entre k y k+1)
    w = 5
    gains = np.abs(np.diff(matrix_values))
    ratios = gains[:-1] / (gains[1:] + 1e-10)

    return ratios


def curvature_method(matrix_values, threshold=0.01):  # Sugerencia: bajar a 0.01 para MEOWA
    """
    Determina el mejor k usando los ratios
    """
    ratios = curvature_ratios(matrix_values)
    best_idx = np.argmax(ratios)
    return best_idx + 2


def tail_ratio(matrix_values, start_k=1, threshold=1.0):
    """
    Implementación del método Tail-Ratio (inspirado en CRB-NCE).
    Compara la ganancia actual con la ganancia máxima de toda la cola futura.
    """
    # 1. Calcular las ganancias absolutas (primeras diferencias)
    # gains[0] es el salto de k=1 a k=2
    # gains[1] es el salto de k=2 a k=3, etc.
    gains = np.abs(np.diff(matrix_values))

    # Protección contra curvas completamente planas
    if np.max(gains) < 1e-10:
        return 2

    ratios = []
    # Calculamos el tail-ratio para cada k
    # Vamos hasta len(gains) - 1 porque el último punto no tiene "cola" futura
    for i in range(len(gains) - 1):
        current_gain = gains[i]

        # La cola es todo lo que viene DESPUÉS de la ganancia actual
        tail_gains = gains[i + 1:]
        tail_max = np.max(tail_gains)

        # Calculamos el ratio (añadimos epsilon para evitar división por cero)
        tr = current_gain / (tail_max + 1e-10)
        ratios.append(tr)

    ratios = np.array(ratios)

    # 2. DECISIÓN: Buscar la primera deceleración permanente
    # Buscamos el primer k donde el Tail-Ratio supera el umbral.
    # Un umbral de 1.0 significa que ninguna ganancia futura superará a la actual.
    valid_indices = np.where(ratios >= threshold)[0]

    if len(valid_indices) > 0:
        # Cogemos el PRIMER índice que cumple la condición
        best_idx = valid_indices[0]
        return best_idx + start_k + 1
    else:
        # Fallback: Si ningún salto es mayor que la cola (curva siempre acelerando),
        # devolvemos el k que tuvo el mejor ratio relativo, o el último evaluable.
        return np.argmax(ratios) + start_k + 1

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
    DB = davies_bouldin_score(X, labels)
    CH = calinski_harabasz_score(X, labels)
    SS = silhouette_score(X, labels)
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


def metrics_basex(U, V, X, y, w):
    xb = Xie_Beni_Index(X, V, U)
    labels = np.argmax(U, axis=1)
    DB = davies_bouldin_score(X, labels)
    CH = calinski_harabasz_score(X, labels)
    results = {"XB": xb, "DB": DB,"CH": CH,
               }
    return results

def metrics_MAOx(U):
    MeM = MAO.MAO_Mean_Max(U)
    results = {
               "MAO_Mean_Max": MeM,
               }

    return results