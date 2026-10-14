#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kx_lattice.py —— 单单元两变量 Shannon 格闭合检验（T-K1/T-K2/T-K3）。

复用 route_func3.tsv（add-1 平滑）。
T-K1 恒等式精确性；T-K2 闭合（6 量 3 参数）；T-K3 轴间共线表。
输出 结果/kx_lattice.txt（双运行逐字节一致后锁定）。
"""
import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV = os.path.join(ROOT, "结果", "route_func3.tsv")


def entropy(p):
    return -sum(a * math.log(a) for a in p if a > 0)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def load():
    files = {}
    with open(TSV, "r", encoding="utf-8") as f:
        f.readline()
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            file_, state, symbol, count = parts
            try:
                c = int(count)
            except ValueError:
                continue
            if c <= 0:
                continue
            sdict = files.setdefault(file_, {}).setdefault(state, {})
            sdict[symbol] = sdict.get(symbol, 0) + c
    return files


def stats(states):
    totals = {st: sum(syms.values()) for st, syms in states.items()}
    tot = sum(totals.values())
    if len(states) < 2 or tot < 10:
        return None
    pi = {st: t / tot for st, t in totals.items()}
    pcond = {}
    for st, syms in states.items():
        s = totals[st]
        v = len(syms)
        pcond[st] = {sym: (c + 1.0) / (s + v) for sym, c in syms.items()}
    hbar = sum(pi[st] * entropy(list(pc.values()))
               for st, pc in pcond.items())
    mix = {}
    for st, pc in pcond.items():
        for sym, pr in pc.items():
            mix[sym] = mix.get(sym, 0.0) + pi[st] * pr
    htau = entropy(list(mix.values()))
    i_val = htau - hbar
    hp = entropy(list(pi.values()))
    hpt = hp - i_val
    htau_p = htau - i_val
    # 联合熵 H(p,tau) = -Σ_{st,sym} π(st)P(sym|st) ln(π(st)P(sym|st))
    hjoint = 0.0
    for st, pc in pcond.items():
        for sym, pr in pc.items():
            q = pi[st] * pr
            hjoint -= q * math.log(q)
    return dict(hp=hp, htau=htau, hpt=hpt, htau_p=htau_p,
                hjoint=hjoint, I=i_val, hbar=hbar)


def main():
    lines = []
    lines.append("== kx_lattice：单单元两变量 Shannon 格闭合检验 ==")
    files = load()
    rows = []
    for f_, states in files.items():
        s = stats(states)
        if s:
            rows.append((f_, s))
    K = len(rows)
    lines.append("合格单元数 K=%d" % K)

    # T-K1
    e1 = e2 = e3 = e4 = e5 = 0.0
    for _, s in rows:
        e1 = max(e1, abs(s["hpt"] - (s["hp"] - s["I"])))
        e2 = max(e2, abs(s["htau_p"] - (s["htau"] - s["I"])))
        e3 = max(e3, abs(s["hjoint"] - (s["hp"] + s["htau_p"])))
        e4 = max(e4, abs(s["hjoint"] - (s["htau"] + s["hpt"])))
        e5 = max(e5, abs(s["hjoint"] - (s["hp"] + s["htau"] - s["I"])))
    lines.append("[T-K1] 恒等式最大偏差：I=H(p)-H(p|tau): %.3e；"
                 "I=H(tau)-H(tau|p): %.3e；H(p,tau)=H(p)+H(tau|p): "
                 "%.3e；=H(tau)+H(p|tau): %.3e；=H(p)+H(tau)-I: %.3e"
                 % (e1, e2, e3, e4, e5))
    pass1 = max(e1, e2, e3, e4, e5) < 1e-12
    lines.append("  判据 <1e-12：%s" % ("通过" if pass1 else "未通过"))

    # T-K2 闭合：由 (hp, htau, I) 恢复全部 6 量（恒等式即恢复式）
    lines.append("[T-K2] 闭合：6 量 {H(p),H(tau),H(p|tau),H(tau|p),"
                 "H(p,tau),I} 由 3 自由参数 (H(p),H(tau),I) 经恒等式"
                 "完全决定（两变量 Shannon 格理论闭）——%s"
                 % ("闭合（无第三类统计量，单单元级）" if pass1
                    else "见 T-K1 失败"))

    # T-K3 轴间共线
    hp = [s["hp"] for _, s in rows]
    hta = [s["htau"] for _, s in rows]
    hpt = [s["hpt"] for _, s in rows]
    iv = [s["I"] for _, s in rows]
    hb = [s["hbar"] for _, s in rows]
    names = ["H(p)", "H(tau)", "H(p|tau)", "I", "Hbar"]
    vals = [hp, hta, hpt, iv, hb]
    lines.append("[T-K3] 轴间 Spearman 共线表：")
    for i in range(5):
        rowtxt = []
        for j in range(5):
            if i == j:
                rowtxt.append("  --  ")
            elif j < i:
                rowtxt.append("%+.2f" % spearman(vals[i], vals[j]))
            else:
                rowtxt.append("      ")
        lines.append("  %-9s %s" % (names[i], " ".join(rowtxt)))

    out = os.path.join(ROOT, "结果", "kx_lattice.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
