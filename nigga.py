import math
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp


def chebyshev(function, n):
    # Define x as a symbolic variable
    x=sp.Symbol('x')

    # Data
    T = [1, x]

    # Chebyshev nodes
    [T.append(sp.expand(2*x*T[i]-T[i-1])) for i in range(1,n)]
    y_i = [function(math.cos((2*i+1)*math.pi/(2*n+2))) for i in range(n+1)]
    sum_y = sum(y_i)
    p_ = (sum_y/(n+1))*T[0]
    p_ += sum((2/(n+1))*sum(y_i[j]*math.cos(((2*j+1)/(2*n+2))*i*math.pi) for j in range(n+1))*T[i] for i in range(1, n+1))
    print(f"p_i = {p_}")

    # Plot
    p_func=sp.lambdify(x,p_)
    X=np.linspace(-1,1,400)
    Y_int = p_func(X)
    Y_og = function(X)
    plt.figure(figsize=(10, 6), dpi=500)
    plt.plot(X, Y_og, label='Função de Runge', color='blue', linewidth=3, alpha=0.4)
    plt.plot(X, Y_int, label=f'Polinómio Chebyshev n={n}', color='red', linewidth=2.5, linestyle='--')
    x_i=[math.cos((2*i+1) * math.pi / (2*(n+1))) for i in range(n+1)]
    plt.scatter(x_i, y_i, color='black', label='Nodos')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolação de Chebyshev da função de Runge')
    plt.grid(True)
    plt.show()

    mp.dps =800
    n_max = 100
    n_values = range(1, n_max + 1)
    errors = []

    x_nodes_max = [mp.cos((2 * k + 1) * mp.pi / (2 * (n_max + 1))) for k in range(n_max + 1)]
    y_nodes_max = [function(xi) for xi in x_nodes_max]

    a_max = []
    for j in range(n_max + 1):
        s = mp.mpf(0)
        for k in range(n_max + 1):
            s += y_nodes_max[k] * mp.cos(j * (2 * k + 1) * mp.pi / (2 * (n_max + 1)))
        coeff = (2 / (n_max + 1)) * s
        if j == 0:
            coeff /= 2
        a_max.append(coeff)
    print(a_max)

    X_dense = [mp.mpf(xi) for xi in np.linspace(-1, 1, 1000)]

    for n in n_values:
        a = a_max[:n + 1]
        max_err = mp.mpf(0)
        for xi in X_dense:
            T0, T1 = mp.mpf(1), xi
            p_val = a[0] * T0 + (a[1] * T1 if n >= 1 else 0)
            for k in range(2, n + 1):
                Tk = 2 * xi * T1 - T0
                p_val += a[k] * Tk
                T0, T1 = T1, Tk
            err = abs(function(xi) - p_val)
            if err > max_err:
                max_err = err
        errors.append(max_err)
        print(f"n = {n}, erro máximo = {max_err}")

    plt.figure(figsize=(10, 6), dpi=500)
    plt.yscale('log')
    plt.semilogy(n_values, errors, marker='o', color='green')
    plt.xlabel('Grau n')
    plt.ylabel('Erro máximo')
    plt.title('Erro máximo da interpolação de Chebyshev')
    plt.grid(True, which='both')
    plt.show()  # exibe o segundo gráfico

def main():
    chebyshev()
main()