"""Maillages : loft étanche et orienté, coque de référence, candidats aléatoires."""
import numpy as np
import pytest
from scipy.stats import qmc

from coque import params as P
from coque.geometrie import loft, maillage
from coque.mesh import CoqueMesh
from coque.ailes import Ailes


def test_loft_boite_volume_exact():
    # boîte 2 x 0.5 x 0.4 lissée sur 5 stations : volume exact, étanche, enroulement cohérent
    X = np.linspace(0, 2, 5)
    rings = np.stack([np.column_stack([X, np.full_like(X, y), np.full_like(X, z)])
                      for y, z in ((0.25, 0.4), (0.25, 0.0), (-0.25, 0.0), (-0.25, 0.4))], axis=1)
    v, f, j = loft(rings)
    m = maillage(v, f)
    assert m.is_watertight and m.is_winding_consistent
    assert m.volume == pytest.approx(2 * 0.5 * 0.4, rel=1e-9)
    assert (j == -1).sum() == 2 * 4            # 4 triangles par bouchon


def test_coque_reference_etanche():
    c = CoqueMesh(**P.separer(P.DEFAUTS)[0])
    m = c.generate()
    assert m.is_watertight and m.is_volume and m.euler_number == 2
    assert m.volume > 0 and m.is_winding_consistent
    assert len(c.face_pont) == len(m.faces) and c.face_pont.any()
    # non-régression géométrique (valeur de la version précédente du générateur)
    assert m.volume == pytest.approx(0.18258, rel=1e-3)
    assert m.bounds[1, 0] == pytest.approx(P.DEFAUTS["L_COQUE"])
    assert m.extents[1] == pytest.approx(P.DEFAUTS["B_MAX"], rel=1e-3)   # stations discrètes


@pytest.mark.parametrize("seed", range(3))
def test_candidats_sobol_etanches(seed):
    X = qmc.Sobol(d=len(P.VARIABLES_LIBRES), scramble=True, seed=seed).random(4)
    n = 0
    for x in X:
        val = P.curseur_vers_valeurs(x)
        if not P.candidat_valide(val)[0]:
            continue
        kc, ka, _ = P.separer(val)
        c = CoqueMesh(**kc)
        m = c.generate()
        assert m.is_watertight and m.volume > 0, val
        a = Ailes.depuis_coque(c, **ka)
        assert a.est_etanche()
        n += 1
    assert n >= 1


def test_ailes_volume_analytique_et_trous():
    c = CoqueMesh(**P.separer(P.DEFAUTS)[0])
    c.generate()
    larg, ep, bord = 0.2, 0.04, 0.03
    a = Ailes.depuis_coque(c, larg, ep, bord)
    L_aile = a.x1 - a.x0
    # section trapézoïdale constante : V = L * largeur * (ep + bord/2)
    assert a.volume(+1) == pytest.approx(L_aile * larg * (ep + bord / 2), rel=1e-6)
    assert a.volume(-1) == pytest.approx(a.volume(+1), rel=1e-9)
    assert a.aire_plan(+1) == pytest.approx(L_aile * larg, rel=1e-6)
    # trous au bord supérieur extérieur, symétriques
    t = a.trous[+1]
    assert t[2] == pytest.approx(c.CREUX) and t[1] > c.B_MAX / 2
    assert np.allclose(a.trous[-1] * [1, -1, 1], t)
    # largeur hors-tout = B_MAX + jeu + largeur d'aile (le bord intérieur suit le flanc)
    assert a.meshes[+1].bounds[1, 1] == pytest.approx(c.B_MAX / 2 + P.AILE_JEU + larg, abs=1e-4)


def test_tumblehome_sans_chevauchement():
    kc = {**P.separer(P.DEFAUTS)[0], "FLARE": -10.0}
    c = CoqueMesh(**kc)
    c.generate()
    a = Ailes.depuis_coque(c, 0.15, 0.10, 0.08)
    # la partie basse de l'aile doit rester hors de la coque : au maître-bau, |y| de la coque
    # à la cote du bas de l'aile est < y_in de l'aile
    z_bas = c.CREUX - 0.18
    i = int(np.argmax(c.B_loc))
    ring = c.anneaux[i]
    y_coque = np.abs(ring[np.abs(ring[:, 2] - z_bas) < 0.01, 1]).max()
    V = a.meshes[+1].vertices
    y_in = V[np.abs(V[:, 0] - c.X[i]) < 0.05, 1].min()      # bord intérieur près du maître-bau
    assert y_in >= y_coque - 1e-3
    assert y_in > c.B_MAX / 2                                  # au-delà du livet (tumblehome)
