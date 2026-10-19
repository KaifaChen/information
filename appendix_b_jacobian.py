#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""appendix_b_jacobian.py — reproduces Appendix B.1 of the paper.

Geometry: three-state circulation family
    T(i -> i+1 mod 3) = a,  T(i -> i-1 mod 3) = b,  T(i -> i) = 1-a-b,
    E(tau=s|s) = w_s,       E(tau != s|s) = (1-w_s)/2,
    theta = (a, b, w0, w1, w2),  pi = (1/3,1/3,1/3) for all (a,b).

Functionals on the family (product convention):
    H(theta)   = (1/3) sum_s [ h(w_s) + (1-w_s) ln 2 ]              (integrand of h_mu)
    I(theta)   = H(Pbar) - (1/3) sum_s H(E(.|s))                    (one-step MI)
    DeltaS     = sum_{s,t} (1/3) T(t|s) KL( E(.|s) || P_b(.|t) )    (piT-weighted directionality)
    A          = sum_s (1/3) [ 1 - sqrt(JS(E(.|s), P_b(.|s)))/ln2 ] (mean same-point consistency)

Outputs:
  1) the Jacobian at theta* = (0.3, 0.1, 0.7, 0.5, 0.3) with singular values, rank, minors;
  2) a multi-point rank check at 20 random family points (seeded, reproducible).
Run: python appendix_b_jacobian.py
"""
import math
import numpy as np

def hbin(w):
    if w <= 0 or w >= 1:
        return 0.0
    return -w * math.log(w) - (1 - w) * math.log(1 - w)

def ent(p):
    return -sum(x * math.log(x) for x in p if x > 0)

def kl(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0 and b > 0)

def js(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def build(theta):
    a, b, w0, w1, w2 = theta
    w = [w0, w1, w2]
    T = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        T[i][(i + 1) % 3] += a
        T[i][(i - 1) % 3] += b
        T[i][i] += 1 - a - b
    E = [[w[i] if j == i else (1 - w[i]) / 2 for j in range(3)] for i in range(3)]
    return T, E

def funcs(theta):
    T, E = build(theta)
    Pb = [[sum(T[j][i] * E[i][k] for i in range(3)) for k in range(3)] for j in range(3)]
    Pbar = [sum(E[i][k] for i in range(3)) / 3 for k in range(3)]
    H = sum(ent(E[s]) for s in range(3)) / 3
    I = ent(Pbar) - H
    dS = sum(T[s][t] * kl(E[s], Pb[t]) for s in range(3) for t in range(3)) / 3
    A = sum(1 - math.sqrt(js(E[s], Pb[s])) / math.log(2) for s in range(3)) / 3
    return np.array([H, I, dS, A])

def jac(theta, h=1e-6):
    th = np.array(theta, float)
    J = np.zeros((4, 5))
    for k in range(5):
        tp = th.copy(); tp[k] += h
        tm = th.copy(); tm[k] -= h
        J[:, k] = (funcs(tp) - funcs(tm)) / (2 * h)
    return J

def main():
    th0 = np.array([0.3, 0.1, 0.7, 0.5, 0.3])
    J = jac(th0)
    names = ["H", "I", "DeltaS", "A"]
    print("theta* =", th0)
    print("values [H, I, DeltaS, A] =", np.round(funcs(th0), 6))
    print("Jacobian (rows H,I,DeltaS,A; cols a,b,w0,w1,w2):")
    for i, nm in enumerate(names):
        print("  %-6s" % nm, " ".join("%+12.6f" % v for v in J[i]))
    sv = np.linalg.svd(J, compute_uv=False)
    print("singular values:", np.round(sv, 6))
    print("row rank (tol 1e-6):", np.linalg.matrix_rank(J, tol=1e-6))
    print("2x2 minor (DeltaS,A) in (a,b):", round(J[2, 0] * J[3, 1] - J[2, 1] * J[3, 0], 6))
    print("2x2 minor (H,I) in (w0,w1):", round(J[0, 2] * J[1, 3] - J[0, 3] * J[1, 2], 6))

    print("\nmulti-point rank check (20 seeded random points of the family):")
    rng = np.random.default_rng(0)
    ranks = []
    for _ in range(20):
        a = rng.uniform(0.05, 0.4)
        b = rng.uniform(0.05, 0.4)
        if a + b >= 0.9:
            continue
        w = rng.uniform(0.15, 0.85, 3)
        R = jac((a, b, w[0], w[1], w[2]))
        ranks.append(np.linalg.matrix_rank(R, tol=1e-6))
    print("  ranks:", ranks)
    print("  all rank 4:", all(r == 4 for r in ranks), "| n =", len(ranks))

if __name__ == "__main__":
    main()
