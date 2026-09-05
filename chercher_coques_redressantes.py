#!/usr/bin/env python3
"""Balayage Sobol de l'espace de conception : liste les coques auto-redressantes (GO),
classées par score, et exporte les meilleures.

    python3 chercher_coques_redressantes.py --n 256 --out coques_redressantes.csv --top 5

Sortie : CSV (une ligne par candidat : variables + masse, KG, GZ_min, AVS, GZ_max, GM0,
franc-bord, énergie, score, verdict) et export STL/JSON des `top` meilleures coques GO dans
./coques_go/. Reprise : les lignes déjà calculées sont relues.
Sert (1) de liste de coques prêtes à l'emploi, (2) de point de départ pour resolve.py.
"""
import argparse
import csv
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

warnings.filterwarnings("ignore")

from coque import params as P
from coque.objectif import evaluer
from coque.export import exporter

KEYS = list(P.VARIABLES_LIBRES)
COLS = KEYS + ["M_kg", "KG_cm", "T_cm", "franc_bord_cm", "GZ_min_B_cm", "phi_GZmin", "GZ_max_cm",
               "AVS", "GM0_cm", "centre_aire", "phi_inond_tribord", "B_tot_m", "S_pan_m2",
               "prod_Wh_j", "E_Wh_km", "score_prop", "penalite", "score", "go", "raison", "t_s"]


def ligne(valeurs):
    t0 = time.time()
    r = evaluer(valeurs)
    st = r.get("stab", {})
    row = {k: round(valeurs[k], 4) for k in KEYS}
    row.update(go=bool(st.get("go", False)), raison=r.get("raison", ""), score=round(r["score"], 2))
    if "M" in st:
        row.update(M_kg=round(st["M"], 2), KG_cm=round(st["KG"] * 100, 2))
    if "GZ_min_B" in st:
        row["GZ_min_B_cm"] = round(st["GZ_min_B"] * 100, 2)
    if "phi" in st:
        row.update(T_cm=round(st["T"] * 100, 1), franc_bord_cm=round(st["franc_bord"] * 100, 1),
                   phi_GZmin=st["phi_GZmin"], GZ_max_cm=round(st["GZ_max"] * 100, 2),
                   AVS=round(st["AVS"], 1), GM0_cm=round(st["GM0"] * 100, 2),
                   centre_aire=round(st["centre_aire"], 1), phi_inond_tribord=st["phi_inondation"][+1])
    if r["valide"]:
        row.update(B_tot_m=round(r["largeur_hors_tout"], 3), S_pan_m2=round(r["surface_panneaux"], 3),
                   prod_Wh_j=round(r["production_Wh_j"], 0), E_Wh_km=round(r["E_Wh_km"], 3),
                   score_prop=round(r["score_prop"], 2), penalite=round(r["penalite"], 2))
    row["t_s"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128, help="nb de candidats Sobol")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="coques_redressantes.csv")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--dossier", default="coques_go")
    ap.add_argument("--inclure-defaut", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()

    X = qmc.Sobol(d=len(KEYS), scramble=True, seed=a.seed).random(a.n)
    done = 0
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = sum(1 for _ in csv.DictReader(f))
        print(f"reprise : {done} candidats déjà dans {a.out}")
    nouveau = not os.path.exists(a.out)
    cands = ([dict(P.DEFAUTS)] if a.inclure_defaut and done == 0 else []) + \
            [P.curseur_vers_valeurs(x) for x in X[done:]]

    best, n_go = [], 0
    print(f"{a.jobs} processus")
    with open(a.out, "a", newline="") as fh, ProcessPoolExecutor(max_workers=a.jobs) as ex:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if nouveau:
            w.writeheader()
        # map() rend les résultats dans l'ordre : le CSV reste un préfixe contigu des candidats
        for i, (valeurs, row) in enumerate(zip(cands, ex.map(ligne, cands)), start=done + 1):
            w.writerow(row); fh.flush()
            flag = "GO " if row["go"] else "   "
            print(f"[{i:4d}/{len(cands)+done}] {flag} score={row['score']:>8} GZmin={row.get('GZ_min_B_cm','-'):>6} "
                  f"B={valeurs['B_MAX']:.2f}+2x{valeurs['AILE_LARGEUR']:.2f} bombé={valeurs['HAUTEUR_BOMBE']:.3f} "
                  f"lest={valeurs['LEST']:.1f} {row['raison'][:60]}  ({row['t_s']}s)")
            if row["go"]:
                n_go += 1
                best.append((row["score"], i, valeurs))
                best.sort(key=lambda t: -t[0]); best = best[: a.top]
    print(f"\n{n_go} coques GO sur {len(cands)}.")
    for rank, (s, i, valeurs) in enumerate(best, 1):
        res = evaluer(valeurs, pas=P.PAS_GZ_RAPPORT, assiette_libre=True, courbes_AB=True)
        fichiers = exporter(res, a.dossier, f"coque_go_{rank:02d}_score{s:.0f}")
        print(f"  #{rank} score={s:.1f}  {fichiers[0]}")


if __name__ == "__main__":
    main()
