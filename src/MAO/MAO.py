from collections.abc import Callable
import numpy as np

def MAO_Max_Min(U: list[list[float]]) -> float:
    """
    F represents the quality of the alternative chosen under the assumption that the decision maker seeks to
    maximize the minimal guaranteed performance of each alternative.
    It is particularly useful in uncertain environments, where ensuring a minimum level of satisfaction
    across all criteria is more important than achieving high performance in only a few of them.
    This operator verifies the row symmetry condition RS1; however, but it is not globally symmetric.

    F (U) = max_i{min_j u_{ij}}

    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.

    Returns:
    F(U): The value of the selected alternative.
    """
    return np.max(np.min(U,axis=1))

def MAO_Min_Max(U: list[list[float]]) -> float:
    """
    This strategy arises in a context where Uij represents the risk (or some negative assesment)
    associated with choosing alternative ai according to criterion or scenario cj.
     .. math::
        F(U) = min_i({max_j(u_{ij}}))
    This MAO represents the quality of the selected alternative under the assumption that the decision maker
    seeks to minimize the risk in the worst scenario or criteria.
    It is the dual formulation of the Max–Min rule, emphasizing robustness and caution in decision making.
    This operator verifies the row symmetry condition RS1; however, but it is not globally symmetric.
    This function implements the MOA Min-Max Strategy.

    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.

    Returns:
    F(U): The value of the selected alternative.
    """
    return np.min(np.max(U,axis=1))

def MAO_Weighted_Sum(U: list[list[float]], w: list[float]) -> float:
    """
    When criteria have different importance weights, aggregation is performed using a weighted average.
    This is one of the most widely used aggregation methods in MCDM.
    The aggregation function in this case is
    F (U ) = max_i {\sum_{j=1}k  w_j  \cdot u_{ij}}
    This function represents the quality of the chosen alternative under the assumption that
    the decision maker assigns different levels of importance wj to each criterion,
    thereby allowing a more flexible and preference-oriented evaluation.
    If, as usual in the study of weighted means, each weight wj is assumed to remain tied to the j-th
    position of in the aggregation even when the criteria are permuted,
    this operator would only verify the row symmetry condition RS3.

    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.
    w: weights of each criteria.

    Returns:
    F(U): The value of the selected alternative.
    """
    return np.max(np.sum(w*U,axis=1))

def MAO_Max_Max(U: list[list[float]]) -> float:
    """
    The MaxiMax (optimistic) strategy focuses on the best possible performance.
    It is thus appropriate when the decision context rewards high potential or best-case scenarios.
    Indeed, this rule reflects a highly optimistic attitude,
    favoring alternatives that have at least one outstanding criterion.
    The corresponding aggregation function
    F (U) = max_i {max_j u_{ij}}
    represents the quality of the alternative chosen under the assumption that
    the decision maker is optimistic and seeks to maximize the best possible outcome that any criterion can offer.
    Notice that this operator is globally symmetric, since the aggregation process focuses solely
    on determining the maximum degree being aggregated,
    and therefore does not need to respect the matrix structure of the information.

    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.

    Returns:
    F(U): The value of the selected alternative.
    """
    return np.max(np.max(U,axis=1))


def MAO_Mean_Max(U: list[list[float]]) -> float:
    """
    The corresponding aggregation function
    F (U) = 1/n sum_i {max_j u_{ij}}

    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.

    Returns:
    F(U): The value of the selected alternative.
    """
    return np.mean(np.max(U,axis=1))


def MAO_OWA(U: list[list[float]], w:list[float]) -> float:
    """
    The OWA operator generalizes several of the previously mentioned aggregation methods.
    In this case, the aggregation function
    F (U ) = max_i {\sum_{j=1}k  w_j  \cdot u_{i(j)}}

    represents the quality of the alternative chosen under the assumption that the decision maker
    can adjust the degree of optimism or pessimism by tuning the ordered weights wj.
    When the weights are concentrated on the largest values, the decision becomes more optimistic (similar to MaxiMax),
    while uniform weights lead to an averaging behavior, and concentration on the smallest values
    leads to a more pessimistic decision (similar to Max–Min).
    Thus, OWA provides a flexible and unified aggregation framework for modeling different decision attitudes.
    Since permutating the criteria does not affect the order in which the degrees uij are considered
    in the OWA operator, this operator verifies the symmetry condition RS1,
    and  it is globally symmetric only when the weight vector corresponds to
    the maximum operator, i.e. when w1 = 1 and wj = 0 for j != 1.


    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.
    w: ordered weights.

    Returns:
    F(U): The value of the selected alternative.
    """
    return np.max(np.sum(w * np.flip(np.sort(U,axis=1),axis=1),axis=1))


def  OWA(x: list[float], w: list[float]) ->  float:
    """
    The OWA operator ...


    Parameters:
    x: input vector
    w: ordered weights.

    Returns:
    OWA(x,w): ...
    """
    return np.sum(w * np.flip(np.sort(x)))


def F1(U: list[list[float]]) -> float:
    """
    .. math::
        F1(U) = 1/N \sum({max_j(u_{ij}}))


    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.

    Returns:
    F(U): The value of the selected alternative.
    """
    return np.mean(np.max(U,axis=1))

def F_Mean_Einstein(U: list[list[float]]) -> float:
    """
    .. math::
        F(U) = 1/N \sum({1 - \prod_j(1-u_{ij})/\prod_j(u_{ij})}))

        F(U) = \frac{1}{N} \sum_{i=1}^N({1 - \frac{\prod_j(1-u_{ij})}{\prod_j(u_{ij})}})


    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.

    Returns:
    F(U): The value of the selected alternative.
    """

    return np.mean(np.prod(1-U,axis=1) / np.prod(U,axis=1))



def MAO_MEOWA(U: list[list[float]], w:list[float],A: Callable[[list[float]],float] = np.max) -> float:
    """
    The OWA operator ...


    Parameters:
    U: utility_matrix (list of list of floats): A matrix where each row represents an alternative
                                             and each column represents a criterios or scenario.
    w: ordered weights.
    
    A: aggregtion function to aggregate each row to a value representing the individual...
       default = np.max

    Returns:
    F(U): The value of the selected alternative.
    """
    return OWA(np.apply_along_axis(A, 1, U),w)

