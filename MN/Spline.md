# README

## Descrição

Este projeto implementa um algoritmo para calcular e avaliar splines de grau arbitrário usando eliminação gaussiana e substituição regressiva para resolver sistemas lineares. O código é capaz de lidar com splines de grau maior que 3, tornando-o flexível para diversas aplicações de interpolação.

## Funcionalidades

### 1. **Eliminação Gaussiana**

A função `gauss_elimination` transforma uma matriz de coeficientes em uma matriz triangular, facilitando a resolução do sistema linear.

### 2. **Substituição Regressiva**

A função `back_substitution` encontra a solução do sistema linear a partir da matriz triangular superior gerada pela eliminação gaussiana.

### 3. **Inicialização de Matrizes**

A função `initialize_matrices` cria as matrizes necessárias para o sistema de equações da spline, garantindo que as condições de contorno sejam atendidas.

### 4. **Preenchimento de Matrizes**

A função `fill_matrices` preenche as matrizes A e b com os valores necessários a partir dos nós e valores fornecidos.

### 5. **Cálculo de Coeficientes**

A função `calculate_coefficients` calcula os coeficientes da spline a partir dos valores de M, permitindo a construção da spline em um grau definido.

### 6. **Cálculo de Coeficientes para Graus Maiores**

A função `calculate_higher_degree_coeffs` calcula coeficientes para graus maiores que 3 utilizando diferenças divididas.

### 7. **Avaliação da Spline**

A função `spline_evaluate` avalia a spline em um ponto específico, permitindo obter o valor interpolado correspondente.

## Como Usar

### Dependências

Certifique-se de ter as seguintes bibliotecas instaladas:

```bash
pip install numpy pandas matplotlib
```

### Exemplo de Uso

Para usar a funcionalidade de cálculo de coeficientes da spline e avaliá-la, siga o exemplo abaixo:

```python
import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt

# Definindo os nós e valores
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 4, 9, 16, 25])

# Chamando a função para calcular os coeficientes sem especificar o grau
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
```

## Resumo das Funções Principais

- `gauss_elimination(A, b)`: Aplica a eliminação gaussiana.
- `back_substitution(A, b, precision_type)`: Realiza a substituição regressiva.
- `initialize_matrices(n, precision_type)`: Inicializa as matrizes A e b.
- `fill_matrices(A, b, X, Y, degree)`: Preenche as matrizes com os dados necessários.
- `calculate_coefficients(X, Y, M, degree)`: Calcula os coeficientes da spline.
- `calculate_higher_degree_coeffs(X, Y, M, i, j, h, degree)`: Calcula coeficientes para graus superiores a 3.
- `spline_coefficients(X, Y, degree, precision_type, solve_func)`: Calcula os coeficientes da spline.
- `spline_evaluate(coefficients, knots, x)`: Avalia a spline em um ponto x.
