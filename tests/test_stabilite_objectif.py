"""Chaîne complète sur des coques connues : verdict, inondation des ailes, score, pénalités."""
import numpy as np
import pytest

from coque import params as P
from coque.objectif import evaluer, score, SCORE_INVALIDE
from coque.stabilite import penalite

GO_CONNUE = {**P.DEFAUTS, "HAUTEUR_BOMBE": 0.15, "B_MAX": 0.35, "FLARE": -10.0, "LEST": 6.0}

# Optimum CMA-ES du 2026-09 (score grille 728.9) : GM0 réel ~0, aile basse noyée à 12°.
# Il n'existait que grâce au GM0 tiré du 1er point de la grille à 10° (l'aile touchait
# l'eau pile à 10°) : il doit maintenant être rejeté, sur la grille comme au pas fin.
EXPLOIT_GRILLE = {
    "L_COQUE": min(2.4, P.VARIABLES_LIBRES["L_COQUE"][1]), "B_MAX": 0.25, "CREUX": 0.22707, "DEADRISE": 20.707, "FLARE": 14.232,
    "F_BOUCHAIN": 0.87673, "W_BOUCHAIN": 2.0078, "X_MAITRE": 0.69957, "REMPL_AV": 0.3252,
    "REMPL_AR": 0.48529, "B_ETRAVE": 0.29327, "B_TABLEAU": 0.50394, "ROCKER_AV": 0.14995,
    "ROCKER_AR": 0.099853, "HAUTEUR_BOMBE": 0.0, "AILE_LARGEUR": 0.27498,
    "AILE_EPAISSEUR": 0.021941, "AILE_BORD": 0.0, "LEST": 0.36483,
}


@pytest.fixture(scope="module")
def res_go():
    return evaluer(GO_CONNUE, courbes_AB=True)


@pytest.fixture(scope="module")
def res_go_fin():
    """Même coque GO, pas fin (assiette bloquée pour comparer GM0 avec res_go)."""
    return evaluer(GO_CONNUE, pas=P.PAS_GZ_RAPPORT, courbes_AB=True)


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


def test_gm0_independant_de_la_grille(res_go, res_go_fin):
    """GM0 vient d'un équilibre dédié à PHI_GM0, pas du premier pas de la grille."""
    st10, st5 = res_go["stab"], res_go_fin["stab"]
    assert st10["GM0"] == pytest.approx(st5["GM0"], abs=1e-4)
    assert "GM0_grille" in st10 and st10["GM0"] >= P.GM0_MIN


def test_angle_inondation_affine(res_go, res_go_fin):
    """L'angle d'inondation est bissecté entre deux points de grille et respecte le mini."""
    st, st5 = res_go["stab"], res_go_fin["stab"]
    a_sec, a_mouille = st["phi_inondation_bas"], st["phi_inondation"][+1]
    res_grille = P.PAS_GZ_OPTIM / 2 ** P.N_BISSECT_INOND
    assert 0 < a_mouille - a_sec <= res_grille + 1e-9           # intervalle de bissection
    assert a_sec >= P.PHI_INONDATION_MIN + P.MARGE_GRILLE_DEG   # GO sur la grille => marge tenue
    # les deux grilles encadrent le même angle vrai : les intervalles se recoupent
    assert st5["phi_inondation_bas"] < a_mouille + 1e-9 and a_sec < st5["phi_inondation"][+1] + 1e-9
    # un candidat GO sur la grille grossière ne peut pas violer le seuil au pas fin
    assert st5["phi_inondation_bas"] >= P.PHI_INONDATION_MIN


@pytest.mark.parametrize("fixture", ["res_go", "res_go_fin"])
def test_gz_apres_inondation(fixture, request):
    """GZ est évalué juste après la noyade de l'aile basse (aile inactive) et entre dans
    le minimum global GZ_min_tot, sans modifier les tableaux de la grille. Si l'inondation
    est détectée pile sur un point de grille, ce point suffit et rien n'est ajouté."""
    st = request.getfixturevalue(fixture)["stab"]
    a, g_apres = st["phi_inondation"][+1], st["GZ_apres_inondation"][+1]
    sur_grille = np.any(np.isclose(st["phi"], a))
    if sur_grille:
        assert g_apres is None
        g_apres = float(st["GZ"][np.isclose(st["phi"], a)][0])
    else:
        assert g_apres is not None
    grille = st["GZ"][(st["phi"] > 0.5) & (st["phi"] < 179.5)].min()
    assert st["GZ_min_tot"] == pytest.approx(min(grille, g_apres))
    assert len(st["GZ"]) == len(st["phi"])
    # l'aile noyée porte moins : GZ juste après inondation < GZ ailes sèches au même angle
    assert g_apres < np.interp(a, st["phi"], st["GZ_A"])


def test_exploit_grille_rejete():
    """L'ancien optimum (GM0 gonflé par l'aile touchant l'eau à 10°) est NO-GO et pénalisé
    sur la grille grossière ET au pas fin ; les deux verdicts sont cohérents."""
    r10 = evaluer(EXPLOIT_GRILLE)
    r5 = evaluer(EXPLOIT_GRILLE, pas=P.PAS_GZ_RAPPORT, assiette_libre=True)
    for r in (r10, r5):
        st = r["stab"]
        assert r["valide"] and not st["go"], st["raison"]
        assert st["GM0"] < P.GM0_MIN
        assert st["phi_inondation_bas"] < P.PHI_INONDATION_MIN
        assert r["penalite"] > 500 and r["score"] < 0
    assert r10["stab"]["GM0_grille"] > r10["stab"]["GM0"] + 0.02      # l'artefact que voyait l'optimiseur
    assert r10["stab"]["marge"] == P.MARGE_GRILLE and r5["stab"]["marge"] == 0.0
    assert r10["stab"]["marge_deg"] == P.MARGE_GRILLE_DEG and r5["stab"]["marge_deg"] == 0.0


def test_penalite_continue():
    assert penalite({"go": True}) == 0.0
    assert penalite({}) > 0
    p1 = penalite({"go": False, "GZ_min_B": 0.0})
    p2 = penalite({"go": False, "GZ_min_B": -0.01})
    assert p2 > p1 > 0
    assert penalite({"go": False, "GZ_min_B": 0.02, "franc_bord": 0.05}) == pytest.approx(20000 * 0.03)
    # angle d'inondation : 1° de déficit = 20 points
    assert penalite({"go": False, "GZ_min_B": 0.02, "phi_inondation_bas": P.PHI_INONDATION_MIN - 10}) \
        == pytest.approx(200.0)
    # marge angulaire de grille sur l'inondation
    assert penalite({"go": False, "GZ_min_B": 0.02, "phi_inondation_bas": P.PHI_INONDATION_MIN,
                     "marge_deg": 0.625}) == pytest.approx(20000 * 0.001 * 0.625)
    # marge de grille : seuils surcotés
    assert penalite({"go": False, "GZ_min_B": 0.02, "marge": 0.002}) == 0.0
    assert penalite({"go": False, "GZ_min_B": 0.011, "marge": 0.002}) == pytest.approx(20000 * 0.001)


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
