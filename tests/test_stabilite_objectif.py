"""Chaîne complète sur des coques connues : verdict, inondation des ailes, score, pénalités."""
import numpy as np
import pytest

from coque import params as P
from coque.objectif import evaluer, score, SCORE_INVALIDE
from coque.stabilite import penalite

# Coque GO de référence (CMA-ES local du 2026-09-10 sous les critères tenue au vent /
# KG + 1 cm / aire GZ, puis pont bombé et nervure de rive relevés pour être GO sur les deux
# grilles) : GM0 7.4 cm, gîte 9° sous 10 m/s, inondation 29°, GZ_min[90,170] 3.2 cm.
# Copie de references/go_reference.json.
GO_CONNUE = {
    "L_COQUE": 2.0134, "B_MAX": 0.375, "CREUX": 0.3104, "DEADRISE": 20.2702, "FLARE": -9.5238,
    "F_BOUCHAIN": 0.4385, "W_BOUCHAIN": 2.9431, "X_MAITRE": 0.6843, "REMPL_AV": 0.5522,
    "REMPL_AR": 0.658, "B_ETRAVE": 0.1575, "B_TABLEAU": 0.3118, "ROCKER_AV": 0.1303,
    "ROCKER_AR": 0.0853, "HAUTEUR_BOMBE": 0.20, "AILE_LARGEUR": 0.1851, "AILE_EPAISSEUR": 0.0892,
    "AILE_BORD": 0.05, "LEST": 9.981,
}

# Optimum CMA-ES du 2026-09 (score grille 728.9) : GM0 réel ~0, aile basse noyée à 12°.
# Il n'existait que grâce au GM0 tiré du 1er point de la grille à 10° (l'aile touchait
# l'eau pile à 10°) : il doit maintenant être rejeté, sur la grille comme au pas fin.
EXPLOIT_GRILLE = {
    "L_COQUE": min(2.4, P.VARIABLES_LIBRES["L_COQUE"][1]), "B_MAX": 0.25, "CREUX": 0.22707, "DEADRISE": 20.707, "FLARE": 14.232,
    "F_BOUCHAIN": 0.87673, "W_BOUCHAIN": 2.0078, "X_MAITRE": 0.69957, "REMPL_AV": 0.3252,
    "REMPL_AR": 0.48529, "B_ETRAVE": 0.29327, "B_TABLEAU": 0.50394, "ROCKER_AV": 0.14995,
    "ROCKER_AR": 0.099853, "AILE_LARGEUR": 0.27498, "LEST": 0.36483,
    # bornes constructibles relevées depuis : on prend les minima (l'artefact subsiste)
    "HAUTEUR_BOMBE": P.VARIABLES_LIBRES["HAUTEUR_BOMBE"][0],
    "AILE_EPAISSEUR": P.VARIABLES_LIBRES["AILE_EPAISSEUR"][0],
    "AILE_BORD": P.VARIABLES_LIBRES["AILE_BORD"][0],
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
    assert st["AVS"] >= 179.9 and st["dGZ180"] < 0        # AVS sur la courbe robuste (~0 à 180°)
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
        g_apres = float(st["GZ_rob"][np.isclose(st["phi"], a)][0])
    else:
        assert g_apres is not None
    # GZ_min_tot : courbe ROBUSTE sur la grille + points post-inondation des deux ailes
    candidats = [st["GZ_rob"][(st["phi"] > 0.5) & (st["phi"] < 179.5)].min(), g_apres]
    for s in (+1, -1):
        g, ph = st["GZ_apres_inondation"][s], st["phi_inondation"][s]
        if g is not None and 0.5 < ph < 179.5:
            candidats.append(g)
    assert st["GZ_min_tot"] == pytest.approx(min(candidats))
    assert len(st["GZ"]) == len(st["GZ_rob"]) == len(st["phi"])
    # l'aile noyée porte moins : GZ juste après inondation < GZ ailes sèches au même angle
    # (GZ_A nominal, g_apres robuste : l'écart de 1 cm·sin(a) va dans le même sens)
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
    g0 = P.MARGE_GZ_MIN
    assert penalite({"go": False, "GZ_min_B": g0 - 0.01, "franc_bord": P.FRANC_BORD_MINI - 0.03}) \
        == pytest.approx(20000 * 0.04)
    # angle d'inondation : 1° de déficit = 20 points
    assert penalite({"go": False, "GZ_min_B": g0 + 0.01, "phi_inondation_bas": P.PHI_INONDATION_MIN - 10}) \
        == pytest.approx(200.0)
    # marge angulaire de grille sur l'inondation
    assert penalite({"go": False, "GZ_min_B": g0 + 0.01, "phi_inondation_bas": P.PHI_INONDATION_MIN,
                     "marge_deg": 0.625}) == pytest.approx(20000 * 0.001 * 0.625)
    # marge de grille : seuils surcotés
    assert penalite({"go": False, "GZ_min_B": g0 + 0.01, "marge": 0.002}) == 0.0
    assert penalite({"go": False, "GZ_min_B": g0 + 0.001, "marge": 0.002}) == pytest.approx(20000 * 0.001)
    # tenue au vent : gîte au-delà du maxi, et aile basse noyée sous le vent (5° de marge)
    assert penalite({"go": False, "GZ_min_B": g0 + 0.01, "gite_vent_fort": P.GITE_VENT_MAX + 3}) \
        == pytest.approx(60.0)
    assert penalite({"go": False, "GZ_min_B": g0 + 0.01, "gite_vent_fort": 10.0,
                     "phi_inondation_bas": 10.0 + P.MARGE_INOND_VENT - 2.0 + 20.0}) == 0.0
    # gîte 30° sous le vent, aile basse à 32° : 20° de trop de gîte + 3° de marge manquante
    assert penalite({"go": False, "GZ_min_B": g0 + 0.01, "gite_vent_fort": 30.0,
                     "phi_inondation_bas": 32.0}) \
        == pytest.approx(20.0 * (30.0 - P.GITE_VENT_MAX) + 20.0 * (30.0 + P.MARGE_INOND_VENT - 32.0))
    # réserve dynamique : 1 mm.rad de déficit = 20 points
    assert penalite({"go": False, "GZ_min_B": g0 + 0.01, "aire_60": P.AIRE_GZ_MIN - 0.001}) \
        == pytest.approx(20.0)


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


# ---------------------------------------------------------------- tenue au vent (vent.py)
def test_gite_sous_vent_analytique():
    """GZ linéaire (GM0·phi) et bras de gîte pris constant (surfaces latérales seules, petits
    angles) : l'équilibre est phi = bras / GM0. Le vent nul donne 0°, un vent que la courbe
    ne rattrape jamais donne 180° (chavirage)."""
    from coque import vent as W
    M, GM0 = 40.0, 0.10
    phi = np.arange(0.0, 180.0 + 1e-9, 0.25)
    gz = GM0 * np.radians(phi)                 # pas de plafond : rattrape toujours le vent
    A_lat, h_lat = 0.5, 0.15
    V = 8.0
    bras0 = W.bras_de_gite(0.0, V, M, A_lat, h_lat, 0.0, 0.0)
    assert bras0 == pytest.approx(0.5 * P.RHO_AIR * V ** 2 * P.CD_FARDAGE * A_lat * h_lat / (M * P.G))
    g = W.gite_sous_vent(phi, gz, V, M, A_lat, h_lat, 0.0, 0.0)
    # bras(phi) = bras0 cos²(phi) : résolution de GM0·phi = bras0 cos²(phi)
    from scipy.optimize import brentq
    attendu = np.degrees(brentq(lambda p: GM0 * p - bras0 * np.cos(p) ** 2, 1e-6, np.pi / 2))
    assert g == pytest.approx(attendu, abs=0.3)
    assert W.gite_sous_vent(phi, gz, 0.0, M, A_lat, h_lat, 0.0, 0.0) == 0.0
    assert W.gite_sous_vent(phi, np.zeros_like(phi) - 0.001, V, M, A_lat, h_lat, 0.0, 0.0) == 180.0
    # plus de vent -> plus de gîte ; le plateau (A_pont) ajoute de la gîte
    assert W.gite_sous_vent(phi, gz, 12.0, M, A_lat, h_lat, 0.0, 0.0) > g
    assert W.gite_sous_vent(phi, gz, V, M, A_lat, h_lat, 1.0, 0.3) > g


def test_courbe_robuste_et_dense(res_go):
    """GZ_rob = GZ - MARGE_KG sin(phi) ; la courbe densifiée passe par les points de grille,
    par le point GM0 et par le point post-inondation ; l'aire 0-60° est celle de la courbe
    robuste."""
    from coque.stabilite import courbe_dense, aire_gz, robuste
    st = res_go["stab"]
    phi, gz, gzr = st["phi"], st["GZ"], st["GZ_rob"]
    assert np.allclose(gzr, gz - P.MARGE_KG * np.sin(np.radians(phi)))
    assert st["GM0"] < st["GM0_grille"] + 0.02        # GM0 robuste = mesuré - MARGE_KG
    pd, gd = courbe_dense(phi, gzr, st["GM0"], st["phi_inondation"], st["GZ_apres_inondation"])
    assert np.allclose(np.interp(phi, pd, gd), gzr, atol=1e-9)
    assert np.interp(P.PHI_GM0, pd, gd) == pytest.approx(st["GM0"] * np.radians(P.PHI_GM0), abs=1e-9)
    ga = st["GZ_apres_inondation"][+1]
    if ga is not None:
        assert np.interp(st["phi_inondation"][+1], pd, gd) == pytest.approx(ga, abs=1e-9)
    assert st["aire_60"] == pytest.approx(aire_gz(pd, gd))
    assert st["aire_60"] >= P.AIRE_GZ_MIN
    assert 0.0 <= st["gite_vent_moyen"] <= st["gite_vent_fort"] <= P.GITE_VENT_MAX
    assert res_go["facteur_gite"] == pytest.approx(np.cos(np.radians(st["gite_vent_moyen"])))


# ---------------------------------------------------------------- piège ailes sèches
def test_equilibres_stables_analytique():
    from coque.stabilite import equilibres_stables
    phi = np.array([90.0, 100.0, 110.0, 120.0, 170.0, 180.0])
    # GZ < 0 puis >= 0 entre 100° et 110° : équilibre stable interpolé ; 180° stable si GZ(170) < 0
    assert equilibres_stables(phi, np.array([1.0, -1.0, 1.0, 2.0, 1.0, 0.0])) == [105.0]
    assert equilibres_stables(phi, np.array([1.0, -1.0, 3.0, 2.0, -1.0, 0.0])) == [102.5, 180.0]
    # GZ > 0 partout : aucun piège (le passage par zéro à 180° descend, il est instable)
    assert equilibres_stables(phi, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])) == []


def test_piege_ailes_seches(res_go, res_go_fin):
    """La coque de référence a un équilibre stable ailes sèches vers 155° (couchée sur une
    aile) ; les trous y sont franchement sous l'eau, donc l'aile se remplit et le bateau en
    sort. Le verdict porte sur le trou le plus profond des deux ailes."""
    for r in (res_go, res_go_fin):
        st = r["stab"]
        assert st["pieges"], "l'état A doit avoir un équilibre stable entre 90° et 180°"
        assert all(P.PHI_PIEGE_MIN <= p["phi"] <= 180.0 for p in st["pieges"])
        assert st["profondeur_trou_min"] == pytest.approx(min(p["profondeur_trou"] for p in st["pieges"]))
        assert st["profondeur_trou_min"] >= P.PROFONDEUR_TROU_MIN + st["marge"]
        # la position couchée est bien entre 150° et 160° et n'est pas 180° : l'inversion
        # complète est instable ailes sèches comme ailes noyées (pont bombé porteur)
        assert 145.0 < st["pieges"][0]["phi"] < 165.0 and 180.0 not in [p["phi"] for p in st["pieges"]]
        assert st["dGZ180"] < 0 and st["GM0_A"] > 0
    # cohérence grille / pas fin
    assert abs(res_go["stab"]["pieges"][0]["phi"] - res_go_fin["stab"]["pieges"][0]["phi"]) < 3.0


def test_penalite_piege():
    g0 = P.MARGE_GZ_MIN
    base = {"go": False, "GZ_min_B": g0 + 0.01}
    assert penalite({**base, "profondeur_trou_min": P.PROFONDEUR_TROU_MIN + 0.05}) == 0.0
    assert penalite({**base, "profondeur_trou_min": float("inf")}) == 0.0         # pas de piège
    assert penalite({**base, "profondeur_trou_min": P.PROFONDEUR_TROU_MIN - 0.01}) == pytest.approx(200.0)
    assert penalite({**base, "profondeur_trou_min": -0.03}) == pytest.approx(20000 * (P.PROFONDEUR_TROU_MIN + 0.03))
