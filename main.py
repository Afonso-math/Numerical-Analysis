import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import sympy as sp
import math

def chebyshev_pol(function, n):
    x = sp.Symbol('x')

    # Generate Chebyshev polynomials
    T = [1, x]
    [T.append(sp.expand(2 * x * T[i] - T[i - 1])) for i in range(1, n)]

    # Compute nodes
    y_i = [function(np.cos((2*i+1)*math.pi/(2*n+2))) for i in range(n+1)]

    # Compute Chebyshev polynomial
    p_ = (sum(y_i) / (n + 1)) * T[0]
    p_ += sum((2 / (n + 1)) * sum(y_i[j] * math.cos(((2 * j + 1) / (2 * n + 2)) * i * math.pi) for j in range(n + 1)) * T[i] for i in range(1, n + 1))
    return sp.simplify(p_)

def plot(function, p_, x, n):
    mp.dps = 50  # precision

    # Compute Chebyshev nodes using mpmath
    x_nodes = [mp.cos((2*i+1) * mp.pi / (2*(n+1))) for i in range(n+1)]
    y_nodes = [function(float(xi)) for xi in x_nodes]  # function must accept mpf

    # Plot polynomial vs original function
    p_func = sp.lambdify(x, p_, modules="mpmath")
    X_dense = [mp.mpf(xi) for xi in np.linspace(-1, 1, 400)]
    Y_int = [p_func(xi) for xi in X_dense]
    Y_og = [function(float(xi) for xi in X_dense]

    plt.figure(figsize=(10, 6), dpi=500)
    plt.plot([float(xi) for xi in X_dense], [float(yi) for yi in Y_og],
             label='Função f(x)', color='blue', linewidth=3, alpha=0.4)
    plt.plot([float(xi) for xi in X_dense], [float(yi) for yi in Y_int],
             label=f'Polinómio Chebyshev n={n}', color='red', linewidth=2.5, linestyle='--')
    plt.scatter([float(xi) for xi in x_nodes], [float(yi) for yi in y_nodes],
                color='black', label='Nodos')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolação de Chebyshev da função de Runge')
    plt.grid(True)
    plt.show()

    n_max = 100
    errors = []

    x_nodes_max = [mp.cos((2*k+1)*mp.pi/(2*(n_max+1))) for k in range(n_max+1)]
    y_nodes_max = [function(float(xi)) for xi in x_nodes_max]

    a_max = []
    for j in range(n_max+1):
        s = mp.mpf(0)
        for k in range(n_max+1):
            s += y_nodes_max[k] * mp.cos(j * (2*k+1) * mp.pi / (2*(n_max+1)))
        coeff = (2/(n_max+1)) * s
        if j == 0:
            coeff /= 2
        a_max.append(coeff)

    X_dense_max = [mp.mpf(xi) for xi in np.linspace(-1,1,1000)]

    for deg in range(1, n_max+1):
        a = a_max[:deg+1]
        max_err = mp.mpf(0)
        for xi in X_dense_max:
            T0, T1 = mp.mpf(1), xi
            p_val = a[0]*T0 + (a[1]*T1 if deg>=1 else 0)
            for k in range(2, deg+1):
                Tk = 2*xi*T1 - T0
                p_val += a[k]*Tk
                T0, T1 = T1, Tk
            err = abs(function(float(xi)) - p_val)
            if err > max_err:
                max_err = err
        errors.append(max_err)

    plt.figure(figsize=(10,6), dpi=500)
    plt.yscale('log')
    plt.semilogy(range(1, n_max+1), [float(e) for e in errors], marker='o', color='green')
    plt.xlabel('Grau n')
    plt.ylabel('Erro máximo')
    plt.title('Erro máximo da interpolação de Chebyshev')
    plt.grid(True, which='both')
    plt.show()

def main():
    # Define the symbolic variable
    x = sp.Symbol('x')

    # Get the function as a string
    function_str = input("Enter the function in terms of x (e.g: x^2, x...): ")

    degree = int(input("Enter the degree of the chebyshev polynomial: "))
    f_expr = sp.sympify(function_str)
    function = sp.lambdify(x, f_expr, modules=["numpy"])

    polynomial = chebyshev_pol(function, degree)
    plot(function, polynomial, x, degree)

main()