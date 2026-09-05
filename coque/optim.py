"""Optimisation CMA-ES multi-départs du score (objectif.score), avec reprise sur checkpoint.

L'état (run en cours + meilleur résultat) est sauvegardé après chaque génération dans
CHECKPOINT_DIR. Relancer après une interruption reprend là où on s'était arrêté, à
condition que le problème soit le même (mêmes variables, même x0, même budget).
"""
import os
import pickle
import time

import numpy as np
import cma

from .objectif import score
from . import params as P

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
MULTISTART_STATE = os.path.join(CHECKPOINT_DIR, "multistart_state.pkl")
RUN_STATE = os.path.join(CHECKPOINT_DIR, "run_state.pkl")


def _save(path, data: bytes):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _load(path):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read()
    return None


def _score_muet(curseur):
    """Point d'entrée des processus fils."""
    return score(curseur, verbose=False)


def cmaes_un_run(x0, sigma0=0.3, budget=5000, popsize=16, seed=1, verbose=True,
                 resume_es=None, executor=None):
    es = resume_es if resume_es is not None else cma.CMAEvolutionStrategy(
        x0, sigma0, {'bounds': [0.0, 1.0], 'maxfevals': budget, 'popsize': popsize,
                     'verbose': -9, 'seed': seed})
    while not es.stop():
        t0 = time.time()
        X = es.ask()
        S = list(executor.map(_score_muet, X)) if executor else [_score_muet(x) for x in X]
        es.tell(X, [-s for s in S])            # CMA-ES minimise
        _save(RUN_STATE, es.pickle_dumps())
        if verbose:
            dt = time.time() - t0
            print(f"      gen {es.countiter:4d} | evals {es.countevals:5d} | gen_best {max(S):8.2f} | "
                  f"meilleur {-es.result.fbest:8.2f} | sigma {es.sigma:.4f} | {len(X)/dt:4.1f} eval/s")
    return np.array(es.result.xbest), -es.result.fbest


def cmaes_multistart(x0=None, n_runs=16, budget_total=50000, executor=None, verbose=True):
    """Plusieurs départs = protection contre les optima locaux. Retourne (best_x, best_f)."""
    x0 = np.array(P.valeurs_vers_curseur(P.DEFAUTS)) if x0 is None else np.asarray(x0, float)
    budget_par_run = budget_total // n_runs
    setup = dict(variables=list(P.VARIABLES_LIBRES), x0=x0.tolist(), n_runs=n_runs,
                 budget_total=budget_total)

    raw = _load(MULTISTART_STATE)
    state = pickle.loads(raw) if raw else None
    if state is not None and state.get("setup") == setup:
        k_start, best_x, best_f = state["k"], state["best_x"], state["best_f"]
        raw_es = _load(RUN_STATE)
        resume_es = pickle.loads(raw_es) if raw_es else None
        print(f"    reprise : run {k_start+1}/{n_runs}, meilleur score actuel = {best_f:.4f}")
    else:
        if state is not None:
            print("    checkpoint d'un autre problème (variables/x0/budget différents) : ignoré")
        k_start, best_x, best_f, resume_es = 0, None, -np.inf, None

    for k in range(k_start, n_runs):
        try:
            x, v = cmaes_un_run(x0, 0.3, budget_par_run, seed=k + 1, verbose=verbose,
                                resume_es=resume_es, executor=executor)
        except KeyboardInterrupt:
            print("\n    interrompu : progression sauvegardée, relancez pour reprendre.")
            raise
        resume_es = None
        if os.path.exists(RUN_STATE):
            os.remove(RUN_STATE)
        marque = ""
        if v > best_f:
            best_f, best_x, marque = v, x, "  <-- meilleur"
        print(f"    run {k+1}/{n_runs} : score = {v:9.4f}{marque}")
        _save(MULTISTART_STATE, pickle.dumps(dict(k=k + 1, best_x=best_x, best_f=best_f, setup=setup)))

    if os.path.exists(MULTISTART_STATE):
        os.remove(MULTISTART_STATE)
    return best_x, best_f
