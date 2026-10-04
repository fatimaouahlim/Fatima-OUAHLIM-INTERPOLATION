import numpy as np
import matplotlib.pyplot as plt

# functions + their derivatives 
def f1(x): return np.exp(x)
def df1(x): return np.exp(x)

def f2(x): return 1 / (1 + 25 * x**2)
def df2(x): return -50 * x / (1 + 25 * x**2)**2

# name, f, f', ylim for the plots
funcs = [("e^x", f1, df1, (0, 3.2)),
         ("1/(1+25x^2)", f2, df2, (-1, 2))]


# ---------------- lagrange ----------------

def lag_basis(nodes, k):
    # L_k = prod (x - xj) / (xk - xj), j != k
    L = np.poly1d([1.0])
    d = 1.0
    for j in range(len(nodes)):
        if j == k:
            continue
        L = L * np.poly1d([1.0, -nodes[j]])
        d *= nodes[k] - nodes[j]
    return L / d, d

def lag_poly(nodes, y):
    P = np.poly1d([0.0])
    for k in range(len(nodes)):
        L, _ = lag_basis(nodes, k)
        P = P + y[k] * L
    return P

def lag_eval(nodes, y, x):
    res = np.zeros_like(x)
    for k in range(len(nodes)):
        L = np.ones_like(x)
        for j in range(len(nodes)):
            if j != k:
                L = L * (x - nodes[j]) / (nodes[k] - nodes[j])
        res = res + y[k] * L
    return res


# ---------------- newton / hermite ----------------

def div_diff(z, y, dy=None):
    
    m = len(z)
    Q = np.zeros((m, m))
    Q[:, 0] = y
    for j in range(1, m):
        for i in range(j, m):
            if z[i] == z[i - j]:
                Q[i, j] = dy(z[i])
            else:
                Q[i, j] = (Q[i, j-1] - Q[i-1, j-1]) / (z[i] - z[i-j])
    return Q

def newton_poly(z, c):
    P = np.poly1d([c[-1]])
    for k in range(len(c) - 2, -1, -1):
        P = P * np.poly1d([1.0, -z[k]]) + c[k]
    return P

def newton_eval(z, c, x):
    P = c[-1] * np.ones_like(x)
    for k in range(len(c) - 2, -1, -1):
        P = P * (x - z[k]) + c[k]
    return P


# ---------------- printing stuff ----------------

def factor(z):
    if z == 0:
        return "x"
    if z < 0:
        return f"(x + {-z:g})"
    return f"(x - {z:g})"

def product(zs):
    
    s = ""
    i = 0
    while i < len(zs):
        j = i
        while j < len(zs) and zs[j] == zs[i]:
            j += 1
        s += factor(zs[i])
        if j - i > 1:
            s += f"^{j - i}"
        i = j
    return s

def poly_str(P):
    s = ""
    for i, a in enumerate(P.coeffs[::-1]):
        a = round(float(a), 6)
        if a == 0:
            continue
        num = f"{abs(a):.6f}".rstrip("0").rstrip(".")
        if i == 0:
            var = ""
        elif i == 1:
            var = "x"
        else:
            var = f"x^{i}"
        term = num + (" " + var if var else "")
        if s == "":
            s = ("-" if a < 0 else "") + term
        else:
            s += (" - " if a < 0 else " + ") + term
    return s

def newton_str(z, c):
    s = f"{c[0]:.6f}"
    for k in range(1, len(c)):
        sign = "+" if c[k] >= 0 else "-"
        s += f" {sign} {abs(c[k]):.6f}{product(z[:k])}"
    return s

def show_table(z, Q):
    for i in range(len(z)):
        row = "  ".join(f"{Q[i, j]:10.6f}" for j in range(i + 1))
        print(f"{z[i]:>5g} | {row}")


# ---------------- part 1: 3 nodes ----------------

nodes = np.array([-1.0, 0.0, 1.0])

for name, f, df, _ in funcs:
    y = f(nodes)
    print("\n" + "=" * 70)
    print(f"f(x) = {name}      nodes: {nodes}")
    print("values f(x_i) :", np.round(y, 6))

    # lagrange
    print("\n--- LAGRANGE ---")
    for k in range(3):
        Lk, d = lag_basis(nodes, k)
        num = "".join(factor(xj) for j, xj in enumerate(nodes) if j != k)
        print(f"L{k}(x) = {num} / ({d:g})  =  {poly_str(Lk)}")
    print("P(x) = " + " + ".join(f"{y[k]:.6f}*L{k}(x)" for k in range(3)))
    PL = lag_poly(nodes, y)
    print("expanded:  P(x) =", poly_str(PL))

    # newton
    print("\n--- NEWTON (divided differences table) ---")
    Q = div_diff(nodes, y)
    show_table(nodes, Q)
    c = np.diag(Q)
    print("P(x) =", newton_str(nodes, c))
    PN = newton_poly(nodes, c)
    print("expanded:  P(x) =", poly_str(PN))
    print("same polynomial as Lagrange?  max |difference| of coefficients =",
          np.max(np.abs(PN.coeffs - PL.coeffs)))

    # hermite
    print("\n--- HERMITE (each node twice, derivatives on the repeated nodes) ---")
    z = np.repeat(nodes, 2)
    Qh = div_diff(z, f(z), df)
    show_table(z, Qh)
    ch = np.diag(Qh)
    print("H(x) =", newton_str(z, ch))
    PH = newton_poly(z, ch)
    print("expanded:  H(x) =", poly_str(PH))


# ---------------- part 2: more points -> runge ----------------

x = np.linspace(-1, 1, 2000)
ns = [3, 5, 9, 13, 17, 21]

for name, f, df, ylim in funcs:
    print("\n" + "=" * 70)
    print(f"Maximum error for f(x) = {name}   (equally spaced points)")
    print("points   Lagrange   Newton   Hermite")

    plt.figure(figsize=(14, 7))
    for i, n in enumerate(ns):
        nodes = np.linspace(-1, 1, n)
        y = f(nodes)

        lag = lag_eval(nodes, y, x)
        new = newton_eval(nodes, np.diag(div_diff(nodes, y)), x)

        z = np.repeat(nodes, 2)
        her = newton_eval(z, np.diag(div_diff(z, f(z), df)), x)

        err = lambda p: np.max(np.abs(f(x) - p))
        print(f"{n:>5}   {err(lag):9.2e}  {err(new):9.2e}  {err(her):9.2e}")

        plt.subplot(2, 3, i + 1)
        plt.plot(x, f(x), "k", linewidth=2.5, label="f(x)")
        plt.plot(x, new, "r", label="Lagrange = Newton")
        plt.plot(x, her, "g--", label="Hermite")
        plt.plot(nodes, y, "ko", markersize=4)
        plt.ylim(*ylim)
        plt.title(f"{n} points")
        plt.legend(fontsize=7)

    plt.suptitle(f"f(x) = {name}")
    plt.tight_layout()
    plt.savefig("runge_exp.png" if name == "e^x" else "runge_rational.png", dpi=120)

plt.show()
