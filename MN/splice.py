import string
import pandas as pd
import numpy as np
from typing import List, Tuple, Callable, Union
import matplotlib.pyplot as plt

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
        coeffs = [0] * (degree + 1)
        
        coeffs[0] = Y[i] 

        if degree >= 1:
            coeffs[1] = (Y[i + 1] - Y[i]) / h[i] - (h[i] / (degree + 1)) * (2 * M[i] + M[i + 1])
        
        if degree >= 2:
            coeffs[2] = M[i] / 2
            
        if degree >= 3:
            coeffs[3] = (M[i + 1] - M[i]) / (degree * h[i])
        
        for j in range(4, degree + 1):
            coeffs[j] = calculate_higher_degree_coeffs(X, Y, M, i, j, h[i], degree)

        coefficients.append(tuple(coeffs))

    return coefficients

def calculate_higher_degree_coeffs(X: np.ndarray, Y: np.ndarray, M: np.ndarray, i: int, j: int, h: float, degree: int) -> float:
    """
    Calcula coeficientes para graus maiores que 3 utilizando diferenças divididas.

    Args:
        X (ndarray): Vetor dos nós.
        Y (ndarray): Vetor dos valores nos nós.
        M (ndarray): Vetor de derivadas.
        i (int): Índice do nó atual.
        j (int): Grau do coeficiente a ser calculado.
        h (float): Distância entre os nós.
        degree (int): Grau da spline.

    Returns:
        float: Valor do coeficiente calculado.
    """
    if j > degree or j < 0:
        return 0.0  # Se o grau solicitado é inválido, retorna zero

    if j == degree:
        # Caso do maior grau, utilize M e Y para cálculo
        return M[i] / (degree + 1)

    # Caso para calcular os coeficientes utilizando diferenças divididas
    term = 0
    for k in range(j + 1):
        term += ((-1) ** (j - k)) * (M[i + k] if (i + k < len(M)) else 0) / (np.math.factorial(k) * h**(j - k))

    return term

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
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 4, 9, 16, 25])

# Chamando sem especificar o grau, para spline de grau máximo
coefficients, knots = spline_coefficients(X, Y)

# Criar um DataFrame com os coeficientes
letters = list(string.ascii_lowercase)[:len(coefficients)+1]
df_coefficients = pd.DataFrame(coefficients, columns=letters)

# Adicionar uma coluna para o intervalo dos nós correspondente
df_coefficients['Intervalo'] = [f'[{knots[i]}, {knots[i+1]}]' for i in range(len(knots)-1)]

# Mostrar o DataFrame com os coeficientes e seus intervalos
print(df_coefficients)

# Calcular os valores de f_x para todo valor no intervalo de X (com passo 0.25)
step = 0.25
x_values = np.arange(X[0], X[-1] + step, step)
f_x_values = [spline_evaluate(coefficients, knots, x) for x in x_values]

# Plotar usando matplotlib
plt.figure(figsize=(10, 6))
plt.plot(X, Y, 'o', label='Dados originais')
plt.plot(x_values, f_x_values, '-', label='Spline interpolada')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.legend()
plt.title('Interpolação Spline')
plt.grid(True)
plt.show()