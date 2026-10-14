#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""examples_check.py —— 两态机与三态反例的数值核验。

核验论文《论信息的约束方程：四泛函的导出、闭合性与动力学》中的笔算实例：
  §3.4 两态机的 pi、h_mu、H(tau)、I、C_mu；
  §3.3 两态机的后向分布 P_b 与逐转移 DeltaS（含"符号即当前状态"的同步发射情形）；
  §4.3 三态反例的 pi 与 DeltaS(A->C)；
  §4.3 两系统一致性实例 A = 1 - sqrt(JS)/ln2。
输出 examples_check.txt（双运行逐字节一致后锁定）。
仅用 Python 3 标准库。
"""
import math


def entropy(p):
    return -sum(a * math.log(a) for a in p if a > 0)


def kl(p, q):
    return sum(p[i] * math.log(p[i] / q[i])
               for i in range(len(p)) if p[i] > 0 and q[i] > 0)


def js(p, q):
    m = [(p[i] + q[i]) / 2.0 for i in range(len(p))]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    lines = []
    # ---- two-state machine (paper's running example) ----
    T = [[0.7, 0.3], [0.2, 0.8]]
    E = {0: (0.7, 0.3), 1: (0.2, 0.8)}
    pi = (0.4, 0.6)  # solves pi*T = pi, pi0+pi1 = 1
    lines.append("== two-state machine: T=[[.7,.3],[.2,.8]], E=(.7,.3)/(.2,.8) ==")
    h0, h1 = entropy(E[0]), entropy(E[1])
    hmu = pi[0] * h0 + pi[1] * h1
    Pbar = (pi[0] * E[0][0] + pi[1] * E[1][0],
            pi[0] * E[0][1] + pi[1] * E[1][1])
    Htau = entropy(Pbar)
    I = Htau - hmu
    Cmu = entropy(pi)
    lines.append("  pi = (%.4f, %.4f)" % pi)
    lines.append("  H(tau|0) = %.4f ; H(tau|1) = %.4f ; h_mu = %.4f"
                 % (h0, h1, hmu))
    lines.append("  Pbar = (%.4f, %.4f) ; H(tau) = %.4f ; I = %.4f ; C_mu = %.4f"
                 % (Pbar[0], Pbar[1], Htau, I, Cmu))

    def Pb(sprime, Emit):
        tot = [0.0, 0.0]
        for s in (0, 1):
            w = pi[s] * T[s][sprime] / pi[sprime]
            tot[0] += w * Emit[s][0]
            tot[1] += w * Emit[s][1]
        return (tot[0], tot[1])

    pb0 = Pb(0, E)
    pb1 = Pb(1, E)
    dS00 = kl(E[0], pb0)
    dS01 = kl(E[0], pb1)
    dS10 = kl(E[1], pb0)
    dS11 = kl(E[1], pb1)
    lines.append("  P_b(.|0) = (%.4f, %.4f) ; P_b(.|1) = (%.4f, %.4f)"
                 % (pb0[0], pb0[1], pb1[0], pb1[1]))
    lines.append("  P_b(.|0) vs P_f(.|0): %s ; P_b(.|1) vs P_f(.|1): %s"
                 % ("EQUAL" if abs(pb0[0]-E[0][0]) < 1e-12 and abs(pb0[1]-E[0][1]) < 1e-12 else "NOT equal",
                    "EQUAL" if abs(pb1[0]-E[1][0]) < 1e-12 and abs(pb1[1]-E[1][1]) < 1e-12 else "NOT equal"))
    lines.append("  DeltaS(0->1) = %.4f ; DeltaS(1->0) = %.4f"
                 " ; DeltaS(0->0) = %.4f ; DeltaS(1->1) = %.4f"
                 % (dS01, dS10, dS00, dS11))

    # sync emission (symbol = current state)
    Esync = {0: (1.0, 0.0), 1: (0.0, 1.0)}
    pbsync1 = Pb(1, Esync)
    dSsync01 = kl(Esync[0], pbsync1)
    lines.append("  sync emission E=delta: P_b(.|1) = (%.4f, %.4f)"
                 " ; P_b(.|1) vs P_f(.|1): %s ; DeltaS(0->1) = %.4f (= ln 5)"
                 % (pbsync1[0], pbsync1[1],
                    "EQUAL" if abs(pbsync1[0]-Esync[1][0]) < 1e-12 and abs(pbsync1[1]-Esync[1][1]) < 1e-12 else "NOT equal",
                    dSsync01))

    # ---- three-state counterexample ----
    lines.append("== three-state counterexample: T(C|A)=T(C|B)=1,"
                 " T(A|C)=T(B|C)=0.5; E(a|A)=E(b|B)=1, E(c|C)=E(d|C)=0.5 ==")
    pi3 = (0.25, 0.25, 0.5)
    EA = (1.0, 0.0, 0.0, 0.0)   # symbols a,b,c,d
    pbc = (0.5, 0.5, 0.0, 0.0)   # mix of A and B, weights (0.25*1/0.5, 0.25*1/0.5)
    dSAC = kl(EA, pbc)
    lines.append("  pi = (0.25, 0.25, 0.5) ; P_b(.|C) = (0.5, 0.5, 0, 0)")
    lines.append("  DeltaS(A->C) = %.4f (= ln 2)" % dSAC)

    # ---- two-system consistency example ----
    lines.append("== two-system example: A = 1 - sqrt(JS)/ln2 ==")
    EA0, EA1 = (0.7, 0.3), (0.2, 0.8)
    EB0, EB1 = (0.9, 0.1), (0.1, 0.9)
    js0, js1 = js(EA0, EB0), js(EA1, EB1)
    wavg = pi[0] * js0 + pi[1] * js1
    A = 1.0 - math.sqrt(wavg) / math.log(2.0)
    lines.append("  JS0 = %.4f ; JS1 = %.4f ; weighted sqrt(JS) = %.4f ; A = %.4f"
                 % (js0, js1, math.sqrt(wavg), A))

    out = "\n".join(lines) + "\n"
    with open("examples_check.txt", "w", encoding="utf-8") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
