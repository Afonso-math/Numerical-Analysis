import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import sympy as sp

def chebyshev_pol(function, n):
    x = sp.Symbol('x')

    # Compute nodes
    y_i = [function(float(mp.cos((2*i+1)*mp.pi/(2*n+2)))) for i in range(n+1)]

    # Compute coefficients
    c = []
    c.append(mp.mpf(sum(y_i)/(n+1)))
    for k in range(1, n + 1):
        s = mp.mpf(0)
        for j in range(n + 1):
            s += y_i[j] * mp.cos(k * (2 * j + 1) * mp.pi / (2 * (n + 1)))
        c.append(2 * s / (n + 1))
    # Compute Chebyshev polynomial
    p_=0
    for i in range(len(c)):
        p_ += c[i]*sp.cos(i*sp.acos(x))
    return p_

def plot(function, p_, x, n, function_str):
    mp.dps = 200  # precision

    # Compute Chebyshev nodes using mpmath
    x_nodes = [mp.cos((2*i+1) * mp.pi / (2*(n+1))) for i in range(n+1)]
    y_nodes = [function(float(xi)) for xi in x_nodes]  # function must accept mpf

    # Plot polynomial vs original function
    p_func = sp.lambdify(x, p_, modules="mpmath")
    X_dense = [mp.mpf(xi) for xi in np.linspace(-1, 1, 1000)]
    Y_int = [p_func(xi) for xi in X_dense]
    Y_og = [mp.mpf(function(float(xi))) for xi in X_dense]

    Maxerr = max(abs(y1 - y2) for y1, y2 in zip(Y_og, Y_int))
    print(Maxerr)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot([float(xi) for xi in X_dense], [float(yi) for yi in Y_og],
             label=f'Função f(x) = {function_str}', color='blue', linewidth=3, alpha=0.4)
    plt.plot([float(xi) for xi in X_dense], [float(yi) for yi in Y_int],
             label=f'Polinómio Chebyshev n={n}', color='red', linewidth=2.5, linestyle='--')
    plt.scatter([float(xi) for xi in x_nodes], [float(yi) for yi in y_nodes],
                color='black', label='Nodos')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Interpolação de {function_str}')
    plt.grid(True)

    plt.text(
        0.05, 0.95,
        f"Erro máximo = {float(Maxerr):.2e}",
        transform=plt.gca().transAxes,
        fontsize=12,
        #horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

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
    plot(function, polynomial, x, degree, function_str)

main()
