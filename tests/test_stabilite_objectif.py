"""Chaîne complète sur des coques connues : verdict, inondation des ailes, score, pénalités."""
import numpy as np
import pytest

from coque import params as P
from coque.objectif import evaluer, score, SCORE_INVALIDE
from coque.stabilite import penalite

GO_CONNUE = {**P.DEFAUTS, "HAUTEUR_BOMBE": 0.15, "B_MAX": 0.35, "FLARE": -10.0}


@pytest.fixture(scope="module")
def res_go():
    return evaluer(GO_CONNUE, courbes_AB=True)


@pytest.fixture(scope="module")
def res_defaut():
    return evaluer(P.DEFAUTS)


def test_defaut_rejete_par_filtre_rapide(res_defaut):
    st = res_defaut["stab"]
    assert res_defaut["valide"]                    # géométrie et hydro OK ...
    assert not st["go"] and "retournée" in st["raison"]   # ... mais pont plat = retourné stable
    assert st["GZ_172"] < 0
    assert res_defaut["penalite"] > 0 and res_defaut["score"] < res_defaut["score_prop"]


def test_coque_go_connue(res_go):
    st = res_go["stab"]
    assert st["go"], st["raison"]
    assert res_go["penalite"] == 0.0 and res_go["score"] == pytest.approx(res_go["score_prop"])
    assert st["GZ_min_B"] >= P.MARGE_GZ_MIN and st["GM0"] >= P.GM0_MIN
    assert st["franc_bord"] >= P.FRANC_BORD_MINI and st["garde_ailes"] >= 0
    assert st["AVS"] == 180.0 and st["dGZ180"] < 0
    assert st["phi"][0] == 0.0 and st["phi"][-1] == 180.0


def test_inondation_sequentielle(res_go):
    """L'aile basse (tribord pour φ>0) se remplit à un angle fini ; avant, la courbe réelle
    suit l'état A (ailes sèches), après elle rejoint l'état B (ailes noyées)."""
    st = res_go["stab"]
    a = st["phi_inondation"][+1]
    assert a is not None and 0 < a < 90
    phi = st["phi"]
    avant, apres = phi < a, phi >= a
    assert np.allclose(st["GZ"][avant], st["GZ_A"][avant], atol=1e-6)
    # après inondation de l'aile basse, l'aile haute est hors d'eau : la poussée est celle de la coque
    # seule -> courbe réelle = état B, sauf près de 180° où l'aile haute peut replonger sans s'être remplie
    m = apres & (phi <= 150)
    assert np.allclose(st["GZ"][m], st["GZ_B"][m], atol=1e-6)
    # à 172°+ les deux ailes sont noyées : GZ_172 du filtre est cohérent avec la courbe
    assert st["GZ_172"] > 0


def test_penalite_continue():
    assert penalite({"go": True}) == 0.0
    assert penalite({}) > 0
    p1 = penalite({"go": False, "GZ_min_B": 0.0})
    p2 = penalite({"go": False, "GZ_min_B": -0.01})
    assert p2 > p1 > 0
    assert penalite({"go": False, "GZ_min_B": 0.02, "franc_bord": 0.05}) == pytest.approx(20000 * 0.03)


def test_score_invalide_et_curseur():
    x = P.valeurs_vers_curseur(P.DEFAUTS)
    assert P.curseur_vers_valeurs(x) == pytest.approx(P.DEFAUTS)
    # deadrise maximal + coque large et plate = section dégénérée -> SCORE_INVALIDE
    inval = {**P.DEFAUTS, "DEADRISE": 45.0, "B_MAX": 0.50, "CREUX": 0.15}
    assert not P.candidat_valide(inval)[0]
    assert evaluer(inval)["score"] == SCORE_INVALIDE
    assert score(x) > SCORE_INVALIDE


def test_largeur_hors_tout_penalisee():
    trop_large = {**GO_CONNUE, "B_MAX": 0.50, "AILE_LARGEUR": 0.30}     # 1,10 m > 0,80 m
    r = evaluer(trop_large)
    assert r["valide"] and r["penalite"] >= 20000 * 0.30 - 1e-6


def test_energie_compte_les_ailes():
    """Plus d'aile = plus de panneaux (et plus de masse) : la surface doit augmenter."""
    r1 = evaluer({**GO_CONNUE, "AILE_LARGEUR": 0.10})
    r2 = evaluer({**GO_CONNUE, "AILE_LARGEUR": 0.20})
    assert r2["surface_panneaux"] > r1["surface_panneaux"]
    assert r2["M"] > r1["M"]
    assert r2["surface_panneaux"] - r1["surface_panneaux"] == pytest.approx(
        P.TAUX_COUVERTURE * 2 * 0.10 * (P.AILE_X1_FRAC - P.AILE_X0_FRAC) * GO_CONNUE["L_COQUE"], rel=1e-6)
