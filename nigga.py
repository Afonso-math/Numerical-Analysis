import matplotlib.pyplot as plt
import mpmath as mp
import sympy as sp
import time

def chebyshev_coefficients(function, n):
    """Calcula os coeficientes de Chebyshev para uma função."""
    y_i = [function(float(mp.cos((2*i+1)*mp.pi/(2*n+2)))) for i in range(n+1)]
    c = [mp.mpf(sum(y_i)/(n+1))]
    for k in range(1, n+1):
        s = mp.mpf(0)
        for j in range(n+1):
            s += y_i[j] * mp.cos(k * (2*j+1) * mp.pi / (2*(n+1)))
        c.append(2 * s / (n+1))
    return c

def clenshaw_eval(c, x_val):
    n = len(c) - 1
    b_kplus1 = mp.mpf(0)
    b_kplus2 = mp.mpf(0)
    for k in range(n, 0, -1):
        b_k = 2 * x_val * b_kplus1 - b_kplus2 + c[k]
        b_kplus2 = b_kplus1
        b_kplus1 = b_k
    return c[0] + x_val * b_kplus1 - b_kplus2

def direct_eval(c, x_val):
    p_ = mp.mpf(0)
    for i in range(len(c)):
        p_ += c[i] * mp.cos(i * mp.acos(x_val))
    return p_

def measure_times(function, degrees, repetitions=1000):
    mp.dps = 200
    x_val = mp.mpf(0.5)

    times_direct = []
    times_clenshaw = []

    for n in degrees:
        c = chebyshev_coefficients(function, n)

        # Direct sum
        start = time.perf_counter()
        for _ in range(repetitions):
            direct_eval(c, x_val)
        t_direct = (time.perf_counter() - start) / repetitions
        times_direct.append(t_direct)

        # Clenshaw
        start = time.perf_counter()
        for _ in range(repetitions):
            clenshaw_eval(c, x_val)
        t_clenshaw = (time.perf_counter() - start) / repetitions
        times_clenshaw.append(t_clenshaw)

        # Print average time
        print(f"n={n}: Soma cossenos = {t_direct:.6f}s, Clenshaw = {t_clenshaw:.6f}s")

    # Plot time
    plt.figure(figsize=(8,5))
    plt.plot(degrees, times_direct, 'o-', label="Soma de cossenos")
    plt.plot(degrees, times_clenshaw, 's-', label="Clenshaw")
    plt.xlabel("Grau do polinômio (n)")
    plt.ylabel("Tempo médio de execução (s)")
    plt.title("Comparação do tempo de execução dos métodos")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    x = sp.Symbol('x')

    function_str = input("Enter the function in terms of x (e.g: x^2, sin(x), ...): ")
    f_expr = sp.sympify(function_str)
    function = sp.lambdify(x, f_expr, modules=["numpy"])

    # Time for diferent n
    degrees = [100, 500, 1000, 2000, 5000]
    measure_times(function, degrees)

main()
