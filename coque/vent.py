"""Gîte statique sous le vent : équilibre couple de gîte aérodynamique / couple de rappel.

Le vent de travers pousse sur les œuvres mortes (L × franc-bord, plus la tranche des ailes)
et, dès que le bateau gîte, sur le plateau solaire (pont + ailes) qui se présente au vent
comme une voile. Le couple de gîte est pris sur le centre de dérive (~ mi-tirant d'eau) :

    C(phi) = q Cd [ A_lat cos(phi) h_lat + A_pont |sin(phi)| h_pont ] cos(phi)
    q = 1/2 rho_air V^2 ; bras de gîte = C / (M g)

La gîte d'équilibre est le premier angle où GZ(phi) >= bras(phi). Si la courbe GZ ne
rattrape jamais le bras de gîte, le bateau chavire sous ce vent (gîte = 180°).
Deux vents : VENT_MOYEN (gîte moyenne -> production solaire réduite en cos(phi)) et
VENT_GITE (vent de calcul -> contrainte GITE_VENT_MAX et aile basse hors d'eau).
"""
from __future__ import annotations

import math

import numpy as np

from . import params as P


def bras_de_gite(phi_deg, V, M, A_lat, h_lat, A_pont, h_pont):
    """Bras de gîte [m] à la gîte phi (deg, scalaire ou tableau) sous le vent V [m/s]."""
    r = np.radians(phi_deg)
    c, s = np.cos(r), np.abs(np.sin(r))
    q = 0.5 * P.RHO_AIR * V ** 2 * P.CD_FARDAGE
    return q * (A_lat * c * h_lat + A_pont * s * h_pont) * c / (M * P.G)


def gite_sous_vent(phi_dense, gz_dense, V, M, A_lat, h_lat, A_pont, h_pont):
    """Premier angle (deg) où GZ >= bras de gîte ; 180 si jamais (chavirage sous ce vent).
    phi_dense/gz_dense : courbe GZ interpolée finement (deg, m), phi croissant depuis 0."""
    if V <= 0:
        return 0.0
    bras = bras_de_gite(phi_dense, V, M, A_lat, h_lat, A_pont, h_pont)
    d = gz_dense - bras
    idx = np.flatnonzero((d[1:] >= 0) & (phi_dense[1:] > 0))
    if len(idx) == 0:
        return 180.0
    i = int(idx[0])                      # d[i] < 0 <= d[i+1] : interpolation linéaire
    a0, a1, d0, d1 = phi_dense[i], phi_dense[i + 1], d[i], d[i + 1]
    return float(a0 if d1 == d0 else a0 + (a1 - a0) * (-d0) / (d1 - d0))


def surfaces_fardage(coque, ailes, franc_bord, T, a_pont):
    """(A_lat, h_lat, A_pont, h_pont) : surfaces exposées [m2] et hauteurs des centres de
    poussée au-dessus du centre de dérive (mi-tirant d'eau) [m]."""
    A_lat = coque.L_COQUE * franc_bord + (ailes.x1 - ailes.x0) * (ailes.epaisseur + ailes.bord)
    h_lat = 0.5 * franc_bord + 0.5 * T
    h_pont = franc_bord + 0.5 * T
    return A_lat, h_lat, a_pont, h_pont


def periode_roulis(I_roll, M, GM0, facteur_masse_ajoutee=1.5):
    """Période propre de roulis [s] (corps rigide + masse d'eau ajoutée forfaitaire)."""
    if GM0 <= 0:
        return math.inf
    return 2 * math.pi * math.sqrt(facteur_masse_ajoutee * I_roll / (M * P.G * GM0))
