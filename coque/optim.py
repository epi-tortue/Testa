"""Optimisation CMA-ES multi-départs du score (objectif.score), avec reprise sur checkpoint.

L'état (run en cours + meilleur résultat) est sauvegardé après chaque génération dans
CHECKPOINT_DIR. Relancer après une interruption reprend là où on s'était arrêté, à
condition que le problème soit le même (mêmes variables, même x0, même budget).

Chaque run CMA-ES travaille sur la courbe GZ grossière (PAS_GZ_OPTIM, assiette bloquée,
seuils surcotés de MARGE_GRILLE). Son optimum est ensuite REVALIDÉ au pas fin
(PAS_GZ_RAPPORT, assiette libre : exactement ce que voit le rapport) et c'est ce score
fin qui sert à classer les runs. Un optimum qui n'existe que grâce à la grille
grossière ne peut donc plus être retenu.

Deux classements sont tenus : le meilleur score fin tout court (la pénalité étant continue,
un quasi-GO peut y battre un vrai GO) et la meilleure coque GO au pas fin. C'est cette
dernière que resolve.py exporte en priorité.
"""
import os
import pickle
import time

import numpy as np
import cma

from .objectif import score, evaluer, resume, rapport
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


def validation_fine(curseur):
    """Réévaluation d'un optimum dans les conditions du rapport. Retourne le dict complet
    d'objectif.evaluer, augmenté de la clé 'go'."""
    res = evaluer(P.curseur_vers_valeurs(curseur), pas=P.PAS_GZ_RAPPORT, assiette_libre=True)
    res["go"] = bool(res["valide"] and res["stab"].get("go", False))
    return res


def _bloc_run(k, n_runs, info, v_grille, res, marque, best_f, best_run, go_f, go_run, n_go, n_faits):
    """Texte affiché à la fin d'un run : statistiques CMA-ES, décomposition du score fin,
    raisons d'un NO-GO, rapport complet de la coque, classement cumulé."""
    st = res.get("stab", {})
    l = [f"    ── run {k+1}/{n_runs} terminé : {info['evals']} évaluations, {info['iters']} générations, "
         f"{info['duree_s']/60:.1f} min ({info['evals']/max(info['duree_s'],1e-9):.2f} éval/s), "
         f"sigma final {info['sigma']:.4f}, arrêt : {info['stop']}"]
    if res["valide"]:
        l.append(f"       score grille {v_grille:9.2f} | score fin {res['score']:9.2f} = prop {res['score_prop']:.2f} "
                 f"- pénalité {res['penalite']:.2f} | GO={res['go']}{marque}")
        l.append(f"       écart grille/fin {res['score'] - v_grille:+.2f} pts  "
                 f"(assiette libre et pas {P.PAS_GZ_RAPPORT:.0f}° contre {P.PAS_GZ_OPTIM:.0f}° bloquée)")
        if not res["go"]:
            for r in st.get("raison", res.get("raison", "")).split(" ; "):
                l.append(f"       NO-GO : {r}")
    else:
        l.append(f"       score grille {v_grille:9.2f} | score fin {res['score']:9.2f} | REJETÉ : {res['raison']}{marque}")
    l += ["       " + x for x in rapport(res).splitlines()]
    l.append(f"    ── classement après {n_faits} run(s) : meilleur score fin {best_f:.2f} (run {best_run}) ; "
             + (f"meilleur GO {go_f:.2f} (run {go_run})" if go_run else "aucun GO")
             + f" ; runs GO : {n_go}/{n_faits}")
    return "\n".join(l)


def cmaes_un_run(x0, sigma0=0.3, budget=5000, popsize=16, seed=1, verbose=True,
                 resume_es=None, executor=None):
    es = resume_es if resume_es is not None else cma.CMAEvolutionStrategy(
        x0, sigma0, {'bounds': [0.0, 1.0], 'maxfevals': budget, 'popsize': popsize,
                     'verbose': -9, 'seed': seed})
    t_run = time.time()
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
    info = dict(evals=int(es.countevals), iters=int(es.countiter), sigma=float(es.sigma),
                duree_s=time.time() - t_run, stop=", ".join(es.stop().keys()) or "budget")
    return np.array(es.result.xbest), -es.result.fbest, info


def cmaes_multistart(x0=None, n_runs=16, budget_total=50000, executor=None, verbose=True):
    """Plusieurs départs = protection contre les optima locaux.
    Retourne un dict : best_x, best_f (meilleur score de validation fine, GO ou non),
    go_x, go_f (meilleure coque GO au pas fin ; None si aucun run n'en a produit)."""
    x0 = np.array(P.valeurs_vers_curseur(P.DEFAUTS)) if x0 is None else np.asarray(x0, float)
    budget_par_run = budget_total // n_runs
    setup = dict(variables=list(P.VARIABLES_LIBRES), x0=x0.tolist(), n_runs=n_runs,
                 budget_total=budget_total)

    raw = _load(MULTISTART_STATE)
    state = pickle.loads(raw) if raw else None
    if state is not None and state.get("setup") == setup:
        k_start, best_x, best_f = state["k"], state["best_x"], state["best_f"]
        go_x, go_f = state.get("go_x"), state.get("go_f", -np.inf)
        best_run, go_run, n_go = state.get("best_run", 0), state.get("go_run", 0), state.get("n_go", 0)
        raw_es = _load(RUN_STATE)
        resume_es = pickle.loads(raw_es) if raw_es else None
        print(f"    reprise : run {k_start+1}/{n_runs}, meilleur score fin actuel = {best_f:.4f}"
              + (f", meilleur GO = {go_f:.4f}" if go_x is not None else ", aucun GO"))
    else:
        if state is not None:
            print("    checkpoint d'un autre problème (variables/x0/budget différents) : ignoré")
        k_start, best_x, best_f, resume_es = 0, None, -np.inf, None
        go_x, go_f = None, -np.inf
        best_run, go_run, n_go = 0, 0, 0

    for k in range(k_start, n_runs):
        try:
            x, v, info = cmaes_un_run(x0, 0.3, budget_par_run, seed=k + 1, verbose=verbose,
                                      resume_es=resume_es, executor=executor)
        except KeyboardInterrupt:
            print("\n    interrompu : progression sauvegardée, relancez pour reprendre.")
            raise
        resume_es = None
        if os.path.exists(RUN_STATE):
            os.remove(RUN_STATE)
        res = validation_fine(x)
        v_fin, go = res["score"], res["go"]
        marque = ""
        if v_fin > best_f:
            best_f, best_x, best_run, marque = v_fin, x, k + 1, "  <-- meilleur score"
        if go:
            n_go += 1
            if v_fin > go_f:
                go_f, go_x, go_run, marque = v_fin, x, k + 1, marque + "  <-- meilleur GO"
        print(_bloc_run(k, n_runs, info, v, res, marque, best_f, best_run, go_f, go_run, n_go, k + 1)
              if verbose else
              f"    run {k+1}/{n_runs} : score grille = {v:9.4f}  score fin = {v_fin:9.4f}  GO={go}{marque}")
        _save(MULTISTART_STATE, pickle.dumps(dict(k=k + 1, best_x=best_x, best_f=best_f,
                                                  go_x=go_x, go_f=go_f, best_run=best_run,
                                                  go_run=go_run, n_go=n_go, setup=setup)))

    if os.path.exists(MULTISTART_STATE):
        os.remove(MULTISTART_STATE)
    return dict(best_x=best_x, best_f=best_f, go_x=go_x, go_f=go_f)
