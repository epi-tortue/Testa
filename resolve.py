#!/usr/bin/env python3
"""Optimisation CMA-ES multi-départs de la coque S1 (coque + ailes + lest).

    python3 resolve.py --test                  # évalue la coque de référence (params.DEFAUTS)
    python3 resolve.py --test --valeurs x.json # évalue un candidat (dict JSON des variables)
    python3 resolve.py [--runs 16] [--budget 50000] [--workers N] [--out outputs]

L'optimisation peut être interrompue (Ctrl+C) et reprise : l'état est sauvegardé après
chaque génération dans checkpoints/. À la fin : rapport complet, STL (coque, ailes,
ensemble), JSON et tracé GZ dans --out.
"""
import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

from coque import params as P
from coque.objectif import evaluer, rapport
from coque.export import exporter
from coque.optim import cmaes_multistart


def rapport_complet(valeurs, dossier, prefixe):
    res = evaluer(valeurs, pas=P.PAS_GZ_RAPPORT, assiette_libre=True, courbes_AB=True)
    print(rapport(res))
    for f in exporter(res, dossier, prefixe):
        print("  ->", f)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="évaluer un candidat sans optimiser")
    ap.add_argument("--valeurs", help="JSON {variable: valeur} (défaut : params.DEFAUTS)")
    ap.add_argument("--runs", type=int, default=16, help="nombre de départs CMA-ES")
    ap.add_argument("--budget", type=int, default=50000, help="évaluations au total")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS") or os.cpu_count() or 1))
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    valeurs = dict(P.DEFAUTS)
    if a.valeurs:
        with open(a.valeurs) as fh:
            valeurs.update(json.load(fh))

    if a.test:
        rapport_complet(valeurs, a.out, "test")
        return

    print(f"{a.workers} processus, {a.runs} départs, budget {a.budget} évaluations")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        best_x, best_f = cmaes_multistart(P.valeurs_vers_curseur(valeurs), n_runs=a.runs,
                                          budget_total=a.budget, executor=ex)
    best = P.curseur_vers_valeurs(best_x)
    print(f"\nmeilleure coque (score optimisation {best_f:.2f}) :")
    print(json.dumps(best, indent=1))
    rapport_complet(best, a.out, "optim")


if __name__ == "__main__":
    main()
