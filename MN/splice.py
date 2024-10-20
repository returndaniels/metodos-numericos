import numpy as np
from typing import List, Tuple, Callable, Union

def gauss_elimination(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica eliminação gaussiana para transformar a matriz A em uma matriz triangular.

    Args:
        A (ndarray): Matriz dos coeficientes.
        b (ndarray): Vetor dos termos independentes.

    Returns:
        Tuple[ndarray, ndarray]: A matriz triangularizada e o vetor b modificado.
    """
    n = len(b)

    for i in range(n):
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]

    return A, b

def back_substitution(A: np.ndarray, b: np.ndarray, precision_type: np.dtype) -> np.ndarray:
    """
    Aplica substituição regressiva para encontrar a solução do sistema linear.

    Args:
        A (ndarray): Matriz triangular superior dos coeficientes.
        b (ndarray): Vetor dos termos independentes.
        precision_type (dtype): Tipo de precisão para os cálculos.

    Returns:
        ndarray: Vetor solução do sistema linear.
    """
    n = len(b)
    x = np.zeros(n, dtype=precision_type)
    
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] = x[i] / A[i][i]

    return x

def initialize_matrices(n: int, precision_type: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inicializa as matrizes A e b para o sistema de equações da spline.

    Args:
        n (int): Número de intervalos.
        precision_type (dtype): Tipo de precisão para os cálculos.

    Returns:
        Tuple[ndarray, ndarray]: Matrizes A e b inicializadas.
    """
    A = np.zeros((n + 1, n + 1), dtype=precision_type)
    b = np.zeros(n + 1, dtype=precision_type)
    A[0, 0] = 1  # Condição de contorno
    A[n, n] = 1  # Condição de contorno
    return A, b

def fill_matrices(A: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, degree: int) -> None:
    """
    Preenche as matrizes A e b com os valores necessários para o sistema.

    Args:
        A (ndarray): Matriz A a ser preenchida.
        b (ndarray): Vetor b a ser preenchido.
        X (ndarray): Vetor dos nós.
        Y (ndarray): Vetor dos valores nos nós.
        degree (int): Grau da spline.
    """
    h = np.diff(X)
    n = len(X) - 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = ((Y[i + 1] - Y[i]) / h[i] - (Y[i] - Y[i - 1]) / h[i - 1]) * (degree + 1)

def calculate_coefficients(X: np.ndarray, Y: np.ndarray, M: np.ndarray, degree: int) -> List[Tuple[float]]:
    """
    Calcula os coeficientes da spline a partir dos valores M.

    Args:
        X (ndarray): Vetor dos nós.
        Y (ndarray): Vetor dos valores nos nós.
        M (ndarray): Vetor de derivadas.
        degree (int): Grau da spline.

    Returns:
        List[Tuple[float]]: Lista de coeficientes da spline.
    """
    coefficients = []
    h = np.diff(X)
    n = len(X) - 1

    for i in range(n):
        a = Y[i]
        b = (Y[i + 1] - Y[i]) / h[i] - (h[i] / (degree + 1)) * (2 * M[i] + M[i + 1])
        c = M[i] / 2
        d = (M[i + 1] - M[i]) / (degree * h[i])
        coefficients.append((a, b, c, d))

    return coefficients

def spline_coefficients(X: np.ndarray, Y: np.ndarray, degree: Union[int, None] = None, 
                        precision_type: np.dtype = np.float64, 
                        solve_func: Union[Callable[[np.ndarray, np.ndarray], np.ndarray], None] = None) -> Tuple[List[Tuple[float]], np.ndarray]:
    """
    Calcula os coeficientes da spline de grau definido usando um método de resolução de sistema.
    Caso o grau não seja fornecido, calcula a spline de grau máximo possível.
    Se a função de resolução não for fornecida ou não ocorrer, utiliza eliminação gaussiana.

    Args:
        X (ndarray): Vetor dos nós.
        Y (ndarray): Vetor dos valores nos nós.
        degree (int, optional): Grau da spline. Se None, usa o grau máximo.
        precision_type (dtype, optional): Tipo de precisão para os cálculos.
        solve_func (callable, optional): Função para resolver o sistema.

    Returns:
        Tuple[List[Tuple[float]], ndarray]: Lista de coeficientes da spline e os nós X.
    """
    n = len(X) - 1
    if degree is None:
        degree = n

    A, b = initialize_matrices(n, precision_type)
    fill_matrices(A, b, X, Y, degree)

    M = None
    if solve_func is not None:
        try:
            M = solve_func(A.copy(), b.copy())
        except Exception as e:
            print(f"Erro ao usar a função de resolução fornecida: {e}")
            
    if M is None:
        A_triangular, b_modified = gauss_elimination(A.copy(), b.copy())
        M = back_substitution(A_triangular, b_modified, precision_type)

    coefficients = calculate_coefficients(X, Y, M, degree)

    return coefficients, X

def spline_evaluate(coefficients: List[Tuple[float]], knots: np.ndarray, x: float) -> float:
    """
    Avalia a spline em um ponto x para splines de grau arbitrário.

    Args:
        coefficients (list): Lista de tuplas com os coeficientes dos polinômios.
        knots (ndarray): Vetor dos nós.
        x (float): Ponto onde a spline será avaliada.

    Returns:
        float: Valor da spline no ponto x.
    """
    n = len(knots) - 1

    for i in range(n):
        if knots[i] <= x <= knots[i + 1]:
            coeffs = coefficients[i]
            h = x - knots[i]
            spline_value = 0
            for j, coef in enumerate(coeffs):
                spline_value += coef * h**j
            return spline_value
    return None

# Exemplo de uso
X = np.array([1, 2, 3, 4])
Y = np.array([1, 4, 9, 16])

# Chamando sem especificar o grau, para spline de grau máximo
coefficients, knots = spline_coefficients(X, Y)
x_value = 2.5
f_x = spline_evaluate(coefficients, knots, x_value)

print("Coeficientes: ", coefficients)
print(f"Spline avaliada em x={x_value}: {f_x}")
