#!/usr/bin/env python3
"""Trace GZ(φ) (réel, A, B) pour la coque de référence et les meilleures coques GO du balayage.

    python3 trace_top_go.py [--csv coques_redressantes.csv] [--top 2] [--png gz_top_go.png]
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from coque import params as P
from coque.objectif import evaluer, resume
from coque.export import tracer_gz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="coques_redressantes.csv")
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--png", default="gz_top_go.png")
    a = ap.parse_args()

    cas = {"référence (défauts)": dict(P.DEFAUTS)}
    try:
        import pandas as pd
        d = pd.read_csv(a.csv)
        d = d[d.go.astype(str) == "True"].sort_values("score", ascending=False).head(a.top)
        for i, (_, r) in enumerate(d.iterrows()):
            cas[f"GO #{i+1} score {r.score:.0f}"] = {k: float(r[k]) for k in P.VARIABLES_LIBRES}
    except FileNotFoundError:
        print(f"{a.csv} absent : seule la référence est tracée")

    res = {}
    for lab, val in cas.items():
        res[lab] = evaluer(val, pas=P.PAS_GZ_RAPPORT, assiette_libre=True, courbes_AB=True)
        print(f"{lab:28s} {resume(res[lab])}")
    tracer_gz(res, a.png)
    print("->", a.png)


if __name__ == "__main__":
    main()
