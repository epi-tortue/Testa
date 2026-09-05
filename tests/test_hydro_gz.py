"""Validation des intégrateurs sur des formes analytiques AVANT de les croire sur la coque :
boîte et sphère pour l'hydrostatique droite, sphère à G décalé pour la courbe GZ."""
import math

import numpy as np
import pytest
import trimesh

from coque import params as P
from coque.hydro import Carene
from coque.gz import Vessel, PointMass, metrics


def boite(L=2.0, B=0.5, H=0.4):
    m = trimesh.creation.box(extents=[L, B, H])
    m.apply_translation([L / 2, 0, H / 2])          # x de 0 à L, z de 0 à H
    return m


def sphere(r=0.2, sub=4):
    return trimesh.creation.icosphere(subdivisions=sub, radius=r)


def test_carene_boite():
    L, B, H = 2.0, 0.5, 0.4
    masse = 100.0
    h = Carene(boite(L, B, H), masse).resoudre()
    T = masse / P.RHO_EAU / (L * B)
    assert h["T"] == pytest.approx(T, abs=2e-5)
    assert h["volume"] == pytest.approx(masse / P.RHO_EAU, rel=1e-4)
    assert h["S"] == pytest.approx(L * B + 2 * (L + B) * T, rel=1e-3)
    assert h["Lwl"] == pytest.approx(L, rel=1e-6) and h["Bwl"] == pytest.approx(B, rel=1e-6)
    assert h["Cb"] == pytest.approx(1.0, rel=1e-3) and h["Cp"] == pytest.approx(1.0, rel=2e-2)


def test_carene_coule():
    assert Carene(boite(), 10_000.0).resoudre() is None


def test_volume_calotte_sphere():
    """Volume sous un plan vs calotte sphérique analytique. L'icosphère est inscrite dans la
    sphère : son volume est ~0,5 % trop faible, d'où la tolérance ; la coupe elle-même doit
    être exacte, ce que vérifie la dernière assertion (plan au-dessus = volume total)."""
    r = 0.2
    s = sphere(r)
    c = Carene(s, 1.0)
    for hc in (0.05, 0.1, 0.2, 0.3):
        v_th = math.pi * hc ** 2 * (3 * r - hc) / 3
        assert c.volume_sous(-r + hc) == pytest.approx(v_th, rel=1e-2)
    assert c.volume_sous(r + 1e-3) == pytest.approx(abs(s.volume), rel=1e-9)


def test_gz_sphere_G_decale():
    """Sphère : B est toujours sur la verticale du centre, donc GZ(φ) = d·sin φ exactement
    pour un G situé à d sous le centre. Teste rotation, équilibre et signe de GZ."""
    r, d = 0.2, 0.05
    s = sphere(r)
    masse = 0.5 * P.RHO_EAU * abs(s.volume)               # demi-immergée
    v = Vessel([s], [PointMass("G", masse, 0.0, 0.0, -d)])
    for phi in (0.0, 30.0, 90.0, 150.0, 180.0):
        eq = v.equilibrium(phi, free_trim=False)
        assert eq["GZ"] == pytest.approx(d * math.sin(math.radians(phi)), abs=5e-4)
        assert eq["zw"] == pytest.approx(0.0, abs=2e-3)   # demi-immergée : flottaison au centre
    # test d'immersion d'un point du bateau : le pôle bas est immergé, le pôle haut non
    eq = v.equilibrium(0.0, free_trim=False)
    assert v.immerge([0, 0, -r], eq) and not v.immerge([0, 0, r], eq)
    # à 90° tribord en bas : le point tribord (+y) est immergé, bâbord non
    eq = v.equilibrium(90.0, free_trim=False)
    assert v.immerge([0, r, 0], eq) and not v.immerge([0, -r, 0], eq)


def test_gz_courbe_et_metriques_sphere():
    r, d = 0.2, 0.05
    s = sphere(r)
    masse = 0.5 * P.RHO_EAU * abs(s.volume)
    v = Vessel([s], [PointMass("G", masse, 0.0, 0.0, -d)])
    ang = np.arange(0, 181, 15.0)
    c = v.gz_curve(ang, free_trim=False)
    m = metrics(c["phi"], c["GZ"], masse)
    assert m["go"] and m["AVS_deg"] == 180.0
    assert m["GZ_max"] == pytest.approx(d, abs=1e-3) and m["phi_GZmax"] == 90.0
    assert m["GM0_m"] == pytest.approx(d * math.sin(math.radians(15)) / math.radians(15), rel=2e-2)
    assert m["GM_inverted_m"] < 0                         # position retournée instable


def test_vessel_corps_actifs():
    """Un corps inactif ne porte pas : la flottaison monte de sa contribution."""
    b = boite(1.0, 0.4, 0.4)
    flotteur = trimesh.creation.box(extents=[1.0, 0.2, 0.1]); flotteur.apply_translation([0.5, 0.4, 0.05])
    masse = 60.0
    v = Vessel([b, flotteur], [PointMass("G", masse, 0.5, 0.0, 0.1)])
    zA = v.equilibrium(0.0, free_trim=False)["zw"]
    zB = v.equilibrium(0.0, free_trim=False, actifs=[0])["zw"]
    assert zB > zA
    assert zB == pytest.approx(masse / P.RHO_EAU / 0.4, abs=1e-4)
    with pytest.raises(ValueError):
        Vessel([flotteur], [PointMass("G", masse, 0.5, 0.4, 0.05)]).equilibrium(0.0, free_trim=False)
