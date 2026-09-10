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
       AVS, GZ_max. Pas 5° + assiette libre pour les rapports (~20 s).
    3. GM0 mesuré à part par un équilibre à PHI_GM0 (2°), ailes dans l'état du repos :
       indépendant du pas de la grille (sinon GZ(10°), gonflé par l'aile qui touche l'eau,
       donnait un faux GM0).
    4. angle d'inondation de l'aile basse affiné par bissection entre les deux points de
       grille qui l'encadrent ; la contrainte >= PHI_INONDATION_MIN porte sur la BORNE
       BASSE de l'intervalle final (dernier angle où les trous sont émergés), surcotée
       de MARGE_GRILLE_DEG sur la grille grossière.
    5. GZ évalué juste après chaque inondation (borne haute de l'intervalle, aile noyée) :
       c'est là que la courbe chute d'un coup, entre deux points de grille. Ce point entre
       dans le contrôle GZ >= 0 mais pas dans les tableaux de la grille.
    6. Tous les critères portent sur GZ_robuste = GZ - MARGE_KG sin(phi) (KG relevé de 1 cm).
    7. Tenue au vent (vent.py) : gîte statique sous VENT_MOYEN (-> production solaire) et
       sous VENT_GITE (contrainte GITE_VENT_MAX, aile basse hors d'eau avec marge), aire
       sous GZ_robuste de 0 à PHI_AIRE (réserve dynamique).
    8. Piège ailes sèches : équilibres stables de la courbe état A sur [90°,180°] (le bateau
       vient de chavirer, les ailes ne sont pas encore pleines). À chacun, les trous d'une
       aile doivent être immergés d'au moins PROFONDEUR_TROU_MIN, sinon rien ne remplit
       l'aile et le bateau reste couché là.
L'optimiseur travaille sur la grille grossière avec des seuils surcotés de MARGE_GRILLE ;
optim.py revalide chaque optimum au pas fin (assiette libre) et classe sur ce score-là.
"""
from __future__ import annotations

import math

import numpy as np

from .gz import Vessel, metrics, total_mass_and_cg, roll_inertia
from .energie import surface_panneaux
from . import vent as W
from . import params as P

I_COQUE = 0
I_AILE = {+1: 1, -1: 2}          # index des corps flottants dans le Vessel


def construire_vessel(coque, ailes, masses, label="S1") -> Vessel:
    return Vessel([coque.mesh, ailes.meshes[+1], ailes.meshes[-1]], masses, label=label)


def actifs_pour(inondees=()):
    return [I_COQUE] + [I_AILE[s] for s in (+1, -1) if s not in inondees]


# ------------------------------------------------------------------ courbes
def courbe_etat(vessel, angles, etat="B", free_trim=False, theta0=0.0):
    """Courbe GZ à état d'ailes figé : 'A' (sèches) ou 'B' (inondées)."""
    actifs = actifs_pour(() if etat == "A" else (+1, -1))
    rows = [vessel.equilibrium(a, free_trim, actifs, theta0) for a in angles]
    return (np.array([r["phi"] for r in rows]), np.array([r["GZ"] for r in rows]), rows)


def angle_inondation(vessel, ailes, s, a_sec, a_mouille, actifs, free_trim=False,
                     n_iter=P.N_BISSECT_INOND, theta0=0.0):
    """Bissection de l'angle où les trous de l'aile s passent sous l'eau, sachant
    qu'ils sont émergés à a_sec et immergés à a_mouille (ailes actives = `actifs`).
    Retourne (a_sec, a_mouille) : l'intervalle final qui encadre l'angle vrai."""
    for _ in range(n_iter):
        a = 0.5 * (a_sec + a_mouille)
        eq = vessel.equilibrium(a, free_trim, actifs, theta0)
        if vessel.immerge(ailes.trous[s], eq):
            a_mouille = a
        else:
            a_sec = a
    return float(a_sec), float(a_mouille)


def courbe_reelle(vessel, ailes, angles, free_trim=False, affiner=True):
    """Courbe GZ avec inondation séquentielle des ailes. Retourne
    (phi, GZ, rows, phi_inondation, phi_sec, gz_apres) où phi_inondation[s] = angle auquel
    l'aile s s'est remplie (None si jamais), phi_sec[s] = dernier angle où ses trous étaient
    émergés (borne basse, conservative pour la contrainte) et gz_apres[s] = GZ à
    phi_inondation[s] avec l'aile noyée (None sans affinage : le point de grille suffit).
    Si `affiner`, l'intervalle est bissecté entre les deux points de grille qui l'encadrent
    (les équilibres de la grille, eux, restent aux angles demandés).
    L'équilibre au repos (0°) est toujours à assiette libre ; en assiette bloquée, les
    autres angles sont calculés à cette assiette de repos (theta0)."""
    inondees = set()
    phi_inond = {+1: None, -1: None}
    phi_sec = {+1: None, -1: None}
    gz_apres = {+1: None, -1: None}
    rows = []
    a_prec = None
    theta0 = 0.0
    for a in angles:
        while True:
            actifs = actifs_pour(inondees)
            # au repos (0°) l'assiette est toujours libre : franc-bord, garde des ailes et
            # tirant d'eau en dépendent, et ils doivent être les mêmes sur la grille de
            # l'optimiseur (assiette bloquée ailleurs) et dans le rapport (assiette libre)
            eq = vessel.equilibrium(a, free_trim or a == 0.0, actifs, theta0)
            if a == 0.0:
                theta0 = eq["theta"]
            nouvelles = [s for s in (+1, -1)
                         if s not in inondees and vessel.immerge(ailes.trous[s], eq)]
            if not nouvelles:
                break
            for s in nouvelles:
                inondees.add(s)
                phi_inond[s] = float(a)
                phi_sec[s] = float(a_prec) if a_prec is not None else 0.0
                if affiner and a_prec is not None and a > a_prec:
                    phi_sec[s], phi_inond[s] = angle_inondation(vessel, ailes, s, a_prec, a,
                                                                actifs, free_trim, theta0=theta0)
                    if phi_inond[s] < a:
                        gz_apres[s] = vessel.equilibrium(phi_inond[s], free_trim,
                                                         actifs_pour(inondees), theta0)["GZ"]
        eq["inondees"] = frozenset(inondees)
        rows.append(eq)
        a_prec = a
    phi = np.array([r["phi"] for r in rows])
    gz = np.array([r["GZ"] for r in rows])
    return phi, gz, rows, phi_inond, phi_sec, gz_apres


def test_180(vessel, phi_test=172.0):
    """Filtre rapide : GZ à phi_test coque seule (ailes noyées). < 0 => retourné stable."""
    return vessel.equilibrium(phi_test, free_trim=False, actifs=[I_COQUE])["GZ"]


def equilibres_stables(phi, gz):
    """Angles (deg) des équilibres stables d'une courbe GZ : GZ passe de < 0 à >= 0 quand
    phi croît (interpolation linéaire), plus 180° si GZ(180-) < 0 (retourné stable)."""
    out = []
    for k in range(len(phi) - 1):
        if gz[k] < 0 <= gz[k + 1]:
            out.append(float(phi[k] + (phi[k + 1] - phi[k]) * (-gz[k]) / (gz[k + 1] - gz[k])))
    if len(phi) >= 2 and phi[-1] >= 180.0 - 1e-9 and gz[-2] < 0 \
            and not (out and out[-1] >= 180.0 - 1e-6):        # déjà trouvé par interpolation
        out.append(180.0)
    return out


def pieges_ailes_seches(vessel, ailes, angles, free_trim=False, theta0=0.0,
                        phi_min=P.PHI_PIEGE_MIN):
    """Positions où le bateau peut rester coincé ailes pleines d'air.
    Retourne (rows_A, pieges) : équilibres état A sur [phi_min,180] et, pour chaque
    équilibre stable, dict(phi, profondeur_trou) où profondeur_trou [m] = immersion du trou
    le plus profond des deux ailes (> 0 = sous l'eau => l'aile se remplira)."""
    ang = [float(a) for a in angles if a >= phi_min - 1e-9]
    actifs = actifs_pour(())
    rows = [vessel.equilibrium(a, free_trim, actifs, theta0) for a in ang]
    gz = np.array([r["GZ"] for r in rows])
    pieges = []
    for a_eq in equilibres_stables(np.array(ang), gz):
        eq = vessel.equilibrium(a_eq, free_trim, actifs, theta0)
        prof = max(-float(hauteur_sur_eau(ailes.trous[s], eq)[0]) for s in (+1, -1))
        pieges.append(dict(phi=a_eq, profondeur_trou=prof))
    return rows, pieges


def gz_min_plage(phi, gz, plage=P.PLAGE_GZ_MIN):
    m = (phi >= plage[0]) & (phi <= plage[1])
    return float(gz[m].min()), float(phi[m][gz[m].argmin()])


def hauteur_sur_eau(points, eq):
    """Hauteur signée (m, > 0 = émergé) de points du repère bateau au-dessus du plan de
    flottaison de l'équilibre `eq` (gîte + assiette)."""
    p = (eq["R"][:3, :3] @ np.atleast_2d(points).T).T
    n = np.array([math.sin(eq["theta"]), 0.0, math.cos(eq["theta"])])
    return (p - np.array([0.0, 0.0, eq["zw"]])) @ n


def gm0_mesure(vessel, actifs, free_trim=False, phi=P.PHI_GM0, theta0=0.0):
    """GM0 = GZ(phi)/phi par un équilibre dédié à petite gîte (état d'ailes du repos)."""
    return vessel.equilibrium(phi, free_trim, actifs, theta0)["GZ"] / math.radians(phi)


def robuste(phi, gz, marge_kg=P.MARGE_KG):
    """GZ avec KG relevé de marge_kg : GZ - marge_kg sin(phi)."""
    return gz - marge_kg * np.sin(np.radians(phi))


def courbe_dense(phi, gz, gm0, phi_inond, gz_apres, pas=0.25):
    """Courbe GZ interpolée finement sur [0,180] à partir des points de grille, du point
    GM0 (PHI_GM0) et des points post-inondation. Retourne (phi_dense, gz_dense)."""
    pts = [(float(a), float(g)) for a, g in zip(phi, gz)]
    pts.append((P.PHI_GM0, gm0 * math.radians(P.PHI_GM0)))
    for s in (+1, -1):
        if gz_apres.get(s) is not None:
            pts.append((float(phi_inond[s]), float(gz_apres[s])))
    pts.sort()
    xa = np.array([p[0] for p in pts])
    ya = np.array([p[1] for p in pts])
    # doublons d'angle (point post-inondation confondu avec la grille) : garder le plus bas
    xu, idx = np.unique(xa, return_index=True)
    yu = np.array([ya[xa == x].min() for x in xu])
    pd = np.arange(0.0, 180.0 + 1e-9, pas)
    return pd, np.interp(pd, xu, yu)


def aire_gz(phi_dense, gz_dense, phi_max=P.PHI_AIRE):
    """Aire sous la partie positive de GZ de 0 à phi_max [m.rad]."""
    m = phi_dense <= phi_max + 1e-9
    return float(np.trapezoid(np.clip(gz_dense[m], 0.0, None), np.radians(phi_dense[m])))


# ------------------------------------------------------------------ verdict
def verdict_auto_redressement(coque, ailes, masses, pas=P.PAS_GZ_OPTIM, assiette_libre=False,
                              filtre_rapide=True, courbes_AB=False, marge=0.0, marge_deg=0.0):
    """Retourne un dict :
        go        : sur GZ_robuste (KG + MARGE_KG) : GZ > 0 sur ]0,180[, GZ_min[90,170] >=
                    MARGE_GZ_MIN, GM0 >= GM0_MIN, aire 0-PHI_AIRE >= AIRE_GZ_MIN, gîte sous
                    VENT_GITE <= GITE_VENT_MAX et < inondation - MARGE_INOND_VENT ;
                    franc-bord >= FRANC_BORD_MINI, ailes hors d'eau au repos,
                    aile basse inondée à phi >= PHI_INONDATION_MIN (borne basse de la bissection)
        raison    : texte si NO-GO
        M, KG, LCG, T, franc_bord, garde_ailes, GZ_172
        phi, GZ (nominal), GZ_rob, phi_inondation, phi_inondation_bas, GZ_apres_inondation,
        GZ_min_B, phi_GZmin, GZ_max, phi_GZmax, AVS, GM0 (robuste, mesuré à PHI_GM0),
        GM0_grille (pente nominale du 1er pas de grille), dGZ180, aire_pos, centre_aire,
        GZ_min_tot (grille + points post-inondation), aire_60, gite_vent_moyen,
        gite_vent_fort, periode_roulis, pieges (liste {phi, profondeur_trou} des
        équilibres stables ailes sèches sur [90,180]), profondeur_trou_min, marge, marge_deg
        -- toutes les métriques scalaires GZ sont sur la courbe ROBUSTE --
        (+ GZ_A, GZ_B, GM0_A, GZ_max_A, AVS_A si courbes_AB)
    `marge` [m] surcote les seuils GZ/GM0, `marge_deg` [°] le seuil d'inondation
    (grille grossière de l'optimiseur).
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
    try:
        phi, gz, rows, phi_inond, phi_sec, gz_apres = courbe_reelle(v, ailes, ang, assiette_libre)
        e0 = rows[0]
        gm0 = gm0_mesure(v, e0["actifs"], assiette_libre, theta0=e0["theta"]) - P.MARGE_KG
    except ValueError as exc:                    # coule à une gîte (ailes noyées) : rejet
        out["raison"] = f"coule en gîte ({exc})"
        return out
    gm0_grille = (gz[1] - gz[0]) / math.radians(phi[1] - phi[0])
    gz_rob = robuste(phi, gz)
    gz_apres_rob = {s: (None if gz_apres[s] is None else
                        float(robuste(phi_inond[s], gz_apres[s]))) for s in (+1, -1)}
    met = metrics(phi, gz_rob, M)
    gmin, pmin = gz_min_plage(phi, gz_rob)
    phi_bas = 180.0 if phi_sec[+1] is None else phi_sec[+1]
    gz_min_tot = float(gz_rob[(phi > 0.5) & (phi < 179.5)].min())
    for s in (+1, -1):                      # chute de GZ juste après l'inondation
        if gz_apres_rob[s] is not None and 0.5 < phi_inond[s] < 179.5:
            gz_min_tot = min(gz_min_tot, gz_apres_rob[s])
    # au repos avec assiette : tirant d'eau = point le plus bas de la coque, franc-bord au
    # livet du maître-bau, garde des ailes = point le plus bas des caissons (au-dessus de
    # l'eau si > 0)
    T = -float(hauteur_sur_eau(mesh.vertices, e0).min())
    x_m = coque.X_MAITRE * coque.L_COQUE
    franc_bord = float(hauteur_sur_eau([x_m, 0.0, coque.CREUX], e0)[0])
    garde_ailes = min(float(hauteur_sur_eau(ailes.meshes[s].vertices, e0).min()) for s in (+1, -1))

    # tenue au vent et réserve dynamique, sur la courbe robuste densifiée
    pd, gd = courbe_dense(phi, gz_rob, gm0, phi_inond, gz_apres_rob)
    a_pont = surface_panneaux(coque, ailes) / P.TAUX_COUVERTURE
    fard = W.surfaces_fardage(coque, ailes, franc_bord, T, a_pont)
    gite_moy = W.gite_sous_vent(pd, gd, P.VENT_MOYEN, M, *fard)
    gite_fort = W.gite_sous_vent(pd, gd, P.VENT_GITE, M, *fard)
    aire_60 = aire_gz(pd, gd)
    t_roulis = W.periode_roulis(roll_inertia(masses, cg), M, gm0)

    # piège ailes sèches : équilibres stables de l'état A sur [90°,180°]
    try:
        _, pieges = pieges_ailes_seches(v, ailes, ang, assiette_libre, e0["theta"])
    except ValueError as exc:
        out["raison"] = f"coule en gîte, ailes sèches ({exc})"
        return out
    prof_min = min((p["profondeur_trou"] for p in pieges), default=math.inf)

    out.update(phi=phi, GZ=gz, GZ_rob=gz_rob, phi_inondation=phi_inond, phi_inondation_bas=phi_bas,
               GZ_apres_inondation=gz_apres_rob,
               T=T, franc_bord=franc_bord, garde_ailes=garde_ailes,
               GZ_min_B=gmin, phi_GZmin=pmin, GZ_max=met["GZ_max"], phi_GZmax=met["phi_GZmax"],
               AVS=met["AVS_deg"], GM0=gm0, GM0_grille=gm0_grille, dGZ180=met["GM_inverted_m"],
               aire_pos=met["area_m_rad"], centre_aire=met["area_centroid_deg"],
               GZ_min_tot=gz_min_tot, aire_60=aire_60, gite_vent_moyen=gite_moy,
               gite_vent_fort=gite_fort, periode_roulis=t_roulis,
               pieges=pieges, profondeur_trou_min=prof_min,
               marge=marge, marge_deg=marge_deg)

    raisons = []
    if out["GZ_min_tot"] < marge:
        raisons.append(f"GZ<0 sur ]0,180[ (min {out['GZ_min_tot']*100:.1f} cm)")
    if gmin < P.MARGE_GZ_MIN + marge:
        raisons.append(f"GZ_min[{P.PLAGE_GZ_MIN[0]:.0f},{P.PLAGE_GZ_MIN[1]:.0f}]={gmin*100:.1f} cm "
                       f"< {(P.MARGE_GZ_MIN + marge)*100:.1f} cm (AVS {met['AVS_deg']:.0f}°)")
    if gm0 < P.GM0_MIN + marge:
        raisons.append(f"stabilité initiale insuffisante (GM0={gm0*100:.1f} cm à {P.PHI_GM0:.0f}°)")
    if phi_bas < P.PHI_INONDATION_MIN + marge_deg:
        raisons.append(f"aile basse inondée dès {phi_bas:.1f}° < {P.PHI_INONDATION_MIN + marge_deg:.2f}°")
    if gite_fort > P.GITE_VENT_MAX - marge_deg:
        raisons.append(f"gîte {gite_fort:.1f}° sous {P.VENT_GITE:.0f} m/s > {P.GITE_VENT_MAX - marge_deg:.1f}°")
    if phi_bas < gite_fort + P.MARGE_INOND_VENT + marge_deg:
        raisons.append(f"aile basse noyée sous {P.VENT_GITE:.0f} m/s (gîte {gite_fort:.1f}° + "
                       f"{P.MARGE_INOND_VENT:.0f}° > inondation {phi_bas:.1f}°)")
    if aire_60 < P.AIRE_GZ_MIN + marge * math.radians(P.PHI_AIRE):
        raisons.append(f"aire GZ 0-{P.PHI_AIRE:.0f}° = {aire_60*1000:.1f} mm.rad < {P.AIRE_GZ_MIN*1000:.0f}")
    if prof_min < P.PROFONDEUR_TROU_MIN + marge:
        pire = min(pieges, key=lambda p: p["profondeur_trou"])
        raisons.append(f"piège ailes sèches à {pire['phi']:.0f}° (trous à {prof_min*100:+.1f} cm sous l'eau "
                       f"< {(P.PROFONDEUR_TROU_MIN + marge)*100:.1f} cm : l'aile ne se remplit pas)")
    if franc_bord < P.FRANC_BORD_MINI:
        raisons.append(f"franc-bord {franc_bord*100:.1f} cm < {P.FRANC_BORD_MINI*100:.0f} cm")
    if garde_ailes < 0:
        raisons.append(f"ailes dans l'eau au repos ({-garde_ailes*100:.1f} cm)")
    out["go"] = not raisons
    out["raison"] = " ; ".join(raisons)

    if courbes_AB:
        _, gA, _ = courbe_etat(v, ang, "A", assiette_libre, e0["theta"])
        _, gB, _ = courbe_etat(v, ang, "B", assiette_libre, e0["theta"])
        mA, mB = metrics(phi, gA, M), metrics(phi, gB, M)
        out.update(GZ_A=gA, GZ_B=gB, GM0_A=mA["GM0_m"], GZ_max_A=mA["GZ_max"], AVS_A=mA["AVS_deg"],
                   GM0_B=mB["GM0_m"], GZ_max_B=mB["GZ_max"], AVS_B=mB["AVS_deg"])
    return out


def penalite(v, k=20000.0):
    """Pénalité continue (>= 0) : 0 si GO, sinon k × déficit (en m) de chaque critère.
    k = 20000 => 1 cm de déficit = 200 points, à comparer à un score propulsion ~300-600 :
    une coque non redressante est toujours battue par une coque redressante.
    Angles (inondation, gîte sous le vent) : 1° de déficit = 1 mm de GZ = 20 points.
    Aire GZ : 1 mm.rad de déficit = 20 points. Piège ailes sèches : 1 cm d'immersion
    manquante des trous = 200 points.
    Les seuils GZ/GM0 sont surcotés de v["marge"], les seuils d'angle de v["marge_deg"]
    (grille grossière), comme dans le verdict."""
    if v.get("go"):
        return 0.0
    if "GZ_min_B" not in v:
        return 10.0 * k * P.MARGE_GZ_MIN               # non étanche / coule
    marge = v.get("marge", 0.0)
    pen = k * max(0.0, P.MARGE_GZ_MIN + marge - v["GZ_min_B"])
    if "GZ_min_tot" in v:
        pen += k * max(0.0, marge - v["GZ_min_tot"])
    if "GM0" in v:
        pen += k * max(0.0, P.GM0_MIN + marge - v["GM0"])
    mdeg = v.get("marge_deg", 0.0)
    if "phi_inondation_bas" in v:
        pen += k * 0.001 * max(0.0, P.PHI_INONDATION_MIN + mdeg - v["phi_inondation_bas"])
    if "gite_vent_fort" in v:
        pen += k * 0.001 * max(0.0, v["gite_vent_fort"] - (P.GITE_VENT_MAX - mdeg))
        if "phi_inondation_bas" in v:
            pen += k * 0.001 * max(0.0, v["gite_vent_fort"] + P.MARGE_INOND_VENT + mdeg
                                   - v["phi_inondation_bas"])
    if "aire_60" in v:
        pen += k * max(0.0, P.AIRE_GZ_MIN + marge * math.radians(P.PHI_AIRE) - v["aire_60"])
    if "profondeur_trou_min" in v and math.isfinite(v["profondeur_trou_min"]):
        pen += k * max(0.0, P.PROFONDEUR_TROU_MIN + marge - v["profondeur_trou_min"])
    if "franc_bord" in v:
        pen += k * max(0.0, P.FRANC_BORD_MINI - v["franc_bord"])
    if "garde_ailes" in v:
        pen += k * max(0.0, -v["garde_ailes"])
    return pen
