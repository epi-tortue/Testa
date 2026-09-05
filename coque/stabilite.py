"""Auto-redressement d'une coque S1 : courbe GZ 0-180° avec inondation des ailes.

Trois courbes possibles pour un même bateau :
    état A     : ailes sèches (portent) à tous les angles      -> tenue de mer, roulis
    état B     : ailes inondées (ne portent pas) à tous les angles -> critère conservatif
    état RÉEL  : ailes sèches au départ ; une aile devient inondée (et le reste) dès que
                 ses trous passent sous l'eau à l'équilibre. C'est la courbe du CDC §6
                 ("remplissage des ailes au-delà de l'angle d'immersion des trous").
Le verdict porte sur la courbe RÉELLE ; le filtre rapide utilise l'état B à 172°
(position retournée : les deux ailes sont forcément noyées).

Deux niveaux de calcul pour l'optimiseur :
    1. test_180 : 1 équilibre coque seule (~0,4 s). GZ(172°) < 0 -> position retournée
       stable -> rejet immédiat.
    2. courbe réelle 0-180° (pas 10°, assiette bloquée, ~4 s) -> GZ_min sur [90°,170°],
       AVS, GZ_max, GM0. Pas 5° + assiette libre pour les rapports (~20 s).
"""
from __future__ import annotations

import numpy as np

from .gz import Vessel, metrics, total_mass_and_cg
from . import params as P

I_COQUE = 0
I_AILE = {+1: 1, -1: 2}          # index des corps flottants dans le Vessel


def construire_vessel(coque, ailes, masses, label="S1") -> Vessel:
    return Vessel([coque.mesh, ailes.meshes[+1], ailes.meshes[-1]], masses, label=label)


def actifs_pour(inondees=()):
    return [I_COQUE] + [I_AILE[s] for s in (+1, -1) if s not in inondees]


# ------------------------------------------------------------------ courbes
def courbe_etat(vessel, angles, etat="B", free_trim=False):
    """Courbe GZ à état d'ailes figé : 'A' (sèches) ou 'B' (inondées)."""
    actifs = actifs_pour(() if etat == "A" else (+1, -1))
    c = vessel.gz_curve(angles, free_trim=free_trim, actifs=actifs)
    return c["phi"], c["GZ"], c["rows"]


def courbe_reelle(vessel, ailes, angles, free_trim=False):
    """Courbe GZ avec inondation séquentielle des ailes. Retourne
    (phi, GZ, rows, phi_inondation) où phi_inondation[s] = angle auquel l'aile s
    s'est remplie (None si jamais)."""
    inondees = set()
    phi_inond = {+1: None, -1: None}
    rows = []
    for a in angles:
        while True:
            eq = vessel.equilibrium(a, free_trim, actifs_pour(inondees))
            nouvelles = [s for s in (+1, -1)
                         if s not in inondees and vessel.immerge(ailes.trous[s], eq)]
            if not nouvelles:
                break
            for s in nouvelles:
                inondees.add(s)
                phi_inond[s] = float(a)
        eq["inondees"] = frozenset(inondees)
        rows.append(eq)
    phi = np.array([r["phi"] for r in rows])
    gz = np.array([r["GZ"] for r in rows])
    return phi, gz, rows, phi_inond


def test_180(vessel, phi_test=172.0):
    """Filtre rapide : GZ à phi_test coque seule (ailes noyées). < 0 => retourné stable."""
    return vessel.equilibrium(phi_test, free_trim=False, actifs=[I_COQUE])["GZ"]


def gz_min_plage(phi, gz, plage=P.PLAGE_GZ_MIN):
    m = (phi >= plage[0]) & (phi <= plage[1])
    return float(gz[m].min()), float(phi[m][gz[m].argmin()])


# ------------------------------------------------------------------ verdict
def verdict_auto_redressement(coque, ailes, masses, pas=P.PAS_GZ_OPTIM, assiette_libre=False,
                              filtre_rapide=True, courbes_AB=False):
    """Retourne un dict :
        go        : GZ_réel > 0 sur ]0,180[, GZ_min[90,170] >= MARGE_GZ_MIN, GM0 >= GM0_MIN,
                    franc-bord >= FRANC_BORD_MINI, ailes hors d'eau au repos
        raison    : texte si NO-GO
        M, KG, LCG, T, franc_bord, garde_ailes, GZ_172
        phi, GZ, phi_inondation, GZ_min_B, phi_GZmin, GZ_max, phi_GZmax, AVS, GM0, dGZ180,
        aire_pos, centre_aire
        (+ GZ_A, GZ_B, GM0_A, GZ_max_A, AVS_A si courbes_AB)
    Les clés de courbe manquent si le candidat est rejeté avant (non étanche, coule, filtre).
    """
    out = dict(go=False, raison="")
    mesh = coque.mesh
    if not mesh.is_watertight or not ailes.est_etanche():
        out["raison"] = "maillage non étanche"
        return out
    M, cg = total_mass_and_cg(masses)
    out.update(M=M, KG=float(cg[2]), LCG=float(cg[0]))
    if M > P.RHO_EAU * abs(mesh.volume):
        out["raison"] = f"coule ailes inondées ({M:.1f} kg > {P.RHO_EAU*abs(mesh.volume):.1f} kg)"
        return out
    v = construire_vessel(coque, ailes, masses)

    if filtre_rapide:
        g172 = test_180(v)
        out["GZ_172"] = g172
        if g172 < 0:
            out["raison"] = f"position retournée stable (GZ(172°)={g172*100:.1f} cm)"
            out["GZ_min_B"] = g172
            return out

    ang = np.arange(0.0, 180.0 + 1e-9, pas)
    phi, gz, rows, phi_inond = courbe_reelle(v, ailes, ang, assiette_libre)
    met = metrics(phi, gz, M)
    gmin, pmin = gz_min_plage(phi, gz)
    e0 = rows[0]
    z_bas, z_livet = float(mesh.bounds[0, 2]), coque.CREUX
    franc_bord = z_livet - e0["zw"]
    garde_ailes = franc_bord - (ailes.epaisseur + ailes.bord)
    out.update(phi=phi, GZ=gz, phi_inondation=phi_inond, T=e0["zw"] - z_bas,
               franc_bord=franc_bord, garde_ailes=garde_ailes,
               GZ_min_B=gmin, phi_GZmin=pmin, GZ_max=met["GZ_max"], phi_GZmax=met["phi_GZmax"],
               AVS=met["AVS_deg"], GM0=met["GM0_m"], dGZ180=met["GM_inverted_m"],
               aire_pos=met["area_m_rad"], centre_aire=met["area_centroid_deg"],
               GZ_min_tot=float(gz[(phi > 0.5) & (phi < 179.5)].min()))

    raisons = []
    if not met["go"]:
        raisons.append(f"GZ<0 sur ]0,180[ (min {out['GZ_min_tot']*100:.1f} cm)")
    if gmin < P.MARGE_GZ_MIN:
        raisons.append(f"GZ_min[{P.PLAGE_GZ_MIN[0]:.0f},{P.PLAGE_GZ_MIN[1]:.0f}]={gmin*100:.1f} cm "
                       f"< {P.MARGE_GZ_MIN*100:.0f} cm (AVS {met['AVS_deg']:.0f}°)")
    if met["GM0_m"] < P.GM0_MIN:
        raisons.append(f"stabilité initiale insuffisante (GM0={met['GM0_m']*100:.1f} cm)")
    if franc_bord < P.FRANC_BORD_MINI:
        raisons.append(f"franc-bord {franc_bord*100:.1f} cm < {P.FRANC_BORD_MINI*100:.0f} cm")
    if garde_ailes < 0:
        raisons.append(f"ailes dans l'eau au repos ({-garde_ailes*100:.1f} cm)")
    out["go"] = not raisons
    out["raison"] = " ; ".join(raisons)

    if courbes_AB:
        _, gA, _ = courbe_etat(v, ang, "A", assiette_libre)
        _, gB, _ = courbe_etat(v, ang, "B", assiette_libre)
        mA, mB = metrics(phi, gA, M), metrics(phi, gB, M)
        out.update(GZ_A=gA, GZ_B=gB, GM0_A=mA["GM0_m"], GZ_max_A=mA["GZ_max"], AVS_A=mA["AVS_deg"],
                   GM0_B=mB["GM0_m"], GZ_max_B=mB["GZ_max"], AVS_B=mB["AVS_deg"])
    return out


def penalite(v, k=20000.0):
    """Pénalité continue (>= 0) : 0 si GO, sinon k × déficit (en m) de chaque critère.
    k = 20000 => 1 cm de déficit = 200 points, à comparer à un score propulsion ~300-600 :
    une coque non redressante est toujours battue par une coque redressante."""
    if v.get("go"):
        return 0.0
    if "GZ_min_B" not in v:
        return 10.0 * k * P.MARGE_GZ_MIN               # non étanche / coule
    pen = k * max(0.0, P.MARGE_GZ_MIN - v["GZ_min_B"])
    if "GZ_min_tot" in v:
        pen += k * max(0.0, -v["GZ_min_tot"])
    if "GM0" in v:
        pen += k * max(0.0, P.GM0_MIN - v["GM0"])
    if "franc_bord" in v:
        pen += k * max(0.0, P.FRANC_BORD_MINI - v["franc_bord"])
    if "garde_ailes" in v:
        pen += k * max(0.0, -v["garde_ailes"])
    return pen
