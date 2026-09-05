#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epi'Tortue - Comparaison pont plat / pont bombé / toit à deux pans / plan incliné
================================================================================

Question tranchée par ce script
-------------------------------
Sur un bateau autonome dont le CAP est imposé par la route (et non par le soleil),
et qui roule/tangue en permanence, quelle géométrie de pont maximise l'énergie
solaire récoltée par m² de panneau ?

Le point clé, souvent oublié : un plan incliné n'a d'intérêt que si son AZIMUT
pointe vers le soleil. Ici l'azimut du panneau est esclave du cap. La simulation
intègre donc le cap réel le long d'une orthodromie, plus le roulis et le tangage.

Modèle
------
  1. Route      : orthodromie entre waypoints, vitesse constante -> lat/lon/cap(t)
  2. Soleil     : pvlib SPA (position vectorielle, le bateau bouge vraiment)
  3. Ciel clair : Ineichen-Perez (trouble de Linke maritime)
  4. Nuages     : indice de ciel clair kc en chaîne AR(1) (persistance météo),
                  moyenne dépendant de la latitude (Atlantique N. vs alizés)
  5. Décomposition GHI -> DNI/DHI : Erbs
  6. Transposition : Hay-Davies, albédo de mer variable (Briegleb) au lieu
                     de l'albédo terrestre 0,2 par défaut - décisif pour les
                     panneaux quasi verticaux
  7. Assiette   : roulis + tangage sinusoïdaux, moyennés sur le cycle de houle
  8. Optique    : modificateur d'angle d'incidence (ASHRAE) - pénalise les
                  fortes incidences, ce qu'un simple cos(AOI) ignore

Sorties : tableau console, CSV horaire, CSV de synthèse, figure PNG.

Usage :
    python3 pont_solaire.py --route antilles
    python3 pont_solaire.py --route retour --roulis 25 --vitesse 2.0
    python3 pont_solaire.py --route antilles --roulis 0     # mer d'huile

Dépendances : pvlib, pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import pvlib

warnings.filterwarnings("ignore", category=RuntimeWarning)

R_TERRE_KM = 6371.0


def arr(x):
    """pvlib renvoie DataFrame ou dict d'arrays selon le type d'entree."""
    return np.nan_to_num(np.asarray(x, dtype=float))


# =============================================================================
# 1. ROUTE - orthodromie, cap réel
# =============================================================================

ROUTES = {
    # Départ classique Microtransat est -> ouest, dans les alizés
    "antilles": {
        "nom": "Brest -> Antilles (est-ouest, alizés)",
        "waypoints": [(48.35, -4.55), (35.0, -25.0), (16.30, -61.50)],
        "depart": "2027-06-15",
    },
    # Traversée ouest -> est par le nord (route de SeaCharger-like, plus rude)
    "retour": {
        "nom": "Terre-Neuve -> Irlande (ouest-est, Atlantique Nord)",
        "waypoints": [(47.55, -52.70), (50.0, -30.0), (51.50, -9.50)],
        "depart": "2027-06-15",
    },
}


def cap_orthodromique(lat1, lon1, lat2, lon2):
    """Cap initial (degrés, 0=N, sens horaire) de 1 vers 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_TERRE_KM * np.arcsin(np.sqrt(a))


def point_intermediaire(lat1, lon1, lat2, lon2, f):
    """Point à la fraction f de l'orthodromie 1->2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    l1, l2 = np.radians(lon1), np.radians(lon2)
    d = distance_km(lat1, lon1, lat2, lon2) / R_TERRE_KM
    if d < 1e-9:
        return lat1, lon1
    a = np.sin((1 - f) * d) / np.sin(d)
    b = np.sin(f * d) / np.sin(d)
    x = a * np.cos(p1) * np.cos(l1) + b * np.cos(p2) * np.cos(l2)
    y = a * np.cos(p1) * np.sin(l1) + b * np.cos(p2) * np.sin(l2)
    z = a * np.sin(p1) + b * np.sin(p2)
    return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))


def construire_route(waypoints, depart, vitesse_noeuds, pas_min, lacet_deg, rng):
    """Trajectoire horodatée : lat, lon, cap (cap route + lacet parasite)."""
    vitesse_kmh = vitesse_noeuds * 1.852
    segments = [
        distance_km(*waypoints[i], *waypoints[i + 1]) for i in range(len(waypoints) - 1)
    ]
    total_km = sum(segments)
    duree_h = total_km / vitesse_kmh

    temps = pd.date_range(
        pd.Timestamp(depart, tz="UTC"),
        periods=int(duree_h * 60 / pas_min) + 1,
        freq=f"{pas_min}min",
    )
    parcouru = np.linspace(0.0, total_km, len(temps))

    lats, lons, caps = [], [], []
    bornes = np.concatenate([[0.0], np.cumsum(segments)])
    for s in parcouru:
        i = min(np.searchsorted(bornes, s, side="right") - 1, len(segments) - 1)
        f = (s - bornes[i]) / segments[i]
        la, lo = point_intermediaire(*waypoints[i], *waypoints[i + 1], f)
        # cap = direction vers un point legerement en avant sur le segment
        # (en fin de segment, on regarde en arriere pour eviter un point degenere)
        if f + 0.01 <= 1.0:
            la2, lo2 = point_intermediaire(*waypoints[i], *waypoints[i + 1], f + 0.01)
            cap = cap_orthodromique(la, lo, la2, lo2)
        else:
            la0, lo0 = point_intermediaire(*waypoints[i], *waypoints[i + 1], f - 0.01)
            cap = cap_orthodromique(la0, lo0, la, lo)
        lats.append(la)
        lons.append(lo)
        caps.append(cap)

    # Lacet parasite : le bateau ne tient jamais exactement son cap (AR(1))
    lacet = np.zeros(len(temps))
    for i in range(1, len(temps)):
        lacet[i] = 0.7 * lacet[i - 1] + rng.normal(0, 1)
    lacet *= lacet_deg / max(lacet.std(), 1e-9)

    return pd.DataFrame(
        {
            "lat": lats,
            "lon": lons,
            "cap_route": caps,
            "cap": (np.array(caps) + lacet) % 360.0,
            "km_parcourus": parcouru,
        },
        index=temps,
    )


# =============================================================================
# 2. RESSOURCE SOLAIRE - ciel clair, nuages, décomposition
# =============================================================================

def kc_moyen_par_latitude(lat):
    """Indice de ciel clair moyen : ~0.72 dans les alizés, ~0.52 vers 50°N."""
    return np.interp(np.abs(lat), [10, 25, 40, 50, 60], [0.68, 0.72, 0.62, 0.52, 0.47])


def ressource_solaire(route, pas_min, rng, linke=3.0):
    """Position solaire, GHI nuageux, DNI/DHI."""
    sp = pvlib.solarposition.spa_python(
        route.index, route["lat"].values, route["lon"].values
    )
    zen = sp["apparent_zenith"].values

    am_rel = pvlib.atmosphere.get_relative_airmass(zen)
    am_abs = pvlib.atmosphere.get_absolute_airmass(am_rel, 101325.0)
    dni_extra = pvlib.irradiance.get_extra_radiation(route.index).values

    cs = pvlib.clearsky.ineichen(
        zen, am_abs, linke_turbidity=linke, altitude=0.0, dni_extra=dni_extra
    )

    # Nuages : AR(1) sur kc, forte persistance (systèmes météo de plusieurs jours)
    n = len(route)
    phi = 0.5 ** (pas_min / (12 * 60))  # temps de corrélation ~12 h
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    kc_bar = kc_moyen_par_latitude(route["lat"].values)
    kc = np.clip(kc_bar + 0.28 * z, 0.12, 1.02)

    ghi = arr(cs["ghi"]) * kc
    erbs = pvlib.irradiance.erbs(ghi, zen, route.index)

    out = pd.DataFrame(
        {
            "zenith": zen,
            "azimuth_solaire": sp["azimuth"].values,
            "ghi": ghi,
            "dni": arr(erbs["dni"]),
            "dhi": arr(erbs["dhi"]),
            "dni_extra": dni_extra,
            "airmass": np.nan_to_num(am_rel, nan=40.0),
            "kc": kc,
        },
        index=route.index,
    )

    # Albédo de mer (Briegleb) : très bas au zénith, remonte au soleil rasant.
    # Sans cela on surestime lourdement tout panneau incliné ou vertical.
    mu = np.clip(np.cos(np.radians(zen)), 0.02, 1.0)
    a_dir = 0.026 / (mu**1.7 + 0.065) + 0.15 * (mu - 0.1) * (mu - 0.5) * (mu - 1.0)
    a_dir = np.clip(a_dir, 0.02, 0.6)
    faisceau = out["dni"].values * mu
    denom = np.maximum(faisceau + out["dhi"].values, 1e-6)
    out["albedo"] = np.clip(
        (a_dir * faisceau + 0.06 * out["dhi"].values) / denom, 0.03, 0.5
    )
    return out


# =============================================================================
# 3. GÉOMÉTRIE - orientation d'une facette dans le repère terrestre
# =============================================================================

def orientation_facette(inclinaison_b, azimut_b, cap, roulis, tangage):
    """
    Repère bateau : x = tribord, y = étrave, z = haut.
    inclinaison_b / azimut_b : pente de la facette et son azimut DEPUIS L'ÉTRAVE.
    Renvoie (inclinaison, azimut) de la normale dans le repère terrestre.
    """
    t, a = np.radians(inclinaison_b), np.radians(azimut_b)
    x = np.full_like(cap, np.sin(t) * np.sin(a), dtype=float)
    y = np.full_like(cap, np.sin(t) * np.cos(a), dtype=float)
    z = np.full_like(cap, np.cos(t), dtype=float)

    r, p = np.radians(roulis), np.radians(tangage)
    x, z = x * np.cos(r) + z * np.sin(r), -x * np.sin(r) + z * np.cos(r)   # roulis
    y, z = y * np.cos(p) - z * np.sin(p), y * np.sin(p) + z * np.cos(p)    # tangage

    psi = np.radians(cap)
    n_e = x * np.cos(psi) + y * np.sin(psi)
    n_n = -x * np.sin(psi) + y * np.cos(psi)
    n_u = np.clip(z, -1.0, 1.0)

    return np.degrees(np.arccos(n_u)), np.degrees(np.arctan2(n_e, n_n)) % 360.0


def poa_facette(inclinaison, azimut, sol):
    """Irradiance utile (après modificateur d'angle d'incidence) sur une facette."""
    comp = pvlib.irradiance.get_total_irradiance(
        surface_tilt=inclinaison,
        surface_azimuth=azimut,
        solar_zenith=sol["zenith"].values,
        solar_azimuth=sol["azimuth_solaire"].values,
        dni=sol["dni"].values,
        ghi=sol["ghi"].values,
        dhi=sol["dhi"].values,
        dni_extra=sol["dni_extra"].values,
        airmass=sol["airmass"].values,
        albedo=sol["albedo"].values,
        model="haydavies",
    )
    aoi = pvlib.irradiance.aoi(
        inclinaison, azimut, sol["zenith"].values, sol["azimuth_solaire"].values
    )
    iam = pvlib.iam.ashrae(aoi, b=0.05)
    return (
        arr(iam) * arr(comp["poa_direct"])
        + 0.95 * arr(comp["poa_sky_diffuse"])
        + 0.93 * arr(comp["poa_ground_diffuse"])
    )


# =============================================================================
# 4. CONFIGURATIONS DE PONT
# =============================================================================
# Chaque config = liste de facettes (inclinaison, azimut/étrave), surfaces égales.
# Le résultat est donc une irradiance PAR M² DE PANNEAU, à surface installée
# constante : on compare bien des géométries, pas des tailles.

CONFIGS = {
    "Pont plat (0°)": [(0, 0)],
    "Pont bombé 8° (2 pans latéraux)": [(8, 90), (8, 270)],
    "Toit 2 pans 15° (latéral)": [(15, 90), (15, 270)],
    "Toit 2 pans 25° (latéral)": [(25, 90), (25, 270)],
    "Toit 2 pans 15° (longitudinal)": [(15, 0), (15, 180)],
    "Plan incliné 20° vers l'étrave": [(20, 0)],
    "Plan incliné 20° vers tribord": [(20, 90)],
    "Plan incliné 35° vers tribord": [(35, 90)],
    "Flancs verticaux (4 faces)": [(90, 0), (90, 90), (90, 180), (90, 270)],
}

# Bornes théoriques inatteignables sur un bateau non orientable, pour l'échelle.
REFERENCES = {
    "[réf.] 20° plein sud, azimut fixe/terre": ("fixe_terre", 20, 180),
    "[réf.] suiveur 2 axes": ("suiveur", None, None),
}


def simuler(route, sol, roulis_deg, tangage_deg, n_phases=12):
    """Irradiance utile moyenne (W/m²) pour chaque configuration."""
    phases = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)
    cap = route["cap"].values
    res = {}

    for nom, facettes in CONFIGS.items():
        cumul = np.zeros(len(route))
        for inc_b, az_b in facettes:
            for ph in phases:
                # Roulis et tangage : sinusoïdes déphasées, moyennées sur le cycle
                roulis = roulis_deg * np.sin(ph)
                tangage = tangage_deg * np.sin(ph + 1.1)
                inc, az = orientation_facette(inc_b, az_b, cap, roulis, tangage)
                cumul += poa_facette(inc, az, sol)
        res[nom] = cumul / (len(facettes) * n_phases)

    for nom, (genre, inc, az) in REFERENCES.items():
        if genre == "fixe_terre":
            res[nom] = poa_facette(
                np.full(len(route), float(inc)), np.full(len(route), float(az)), sol
            )
        else:  # suiveur 2 axes parfait
            res[nom] = poa_facette(
                sol["zenith"].values, sol["azimuth_solaire"].values, sol
            )

    return pd.DataFrame(res, index=route.index)


# =============================================================================
# 5. RESTITUTION
# =============================================================================

def journalier(poa, pas_min):
    """Énergie quotidienne par m², journées incomplètes écartées."""
    quotidien = poa.resample("1D").sum() * (pas_min / 60) / 1000.0
    complet = poa.resample("1D").size() == int(24 * 60 / pas_min)
    return quotidien[complet.values]


def synthese(poa, poa_inverse, pas_min):
    """
    Énergie journalière par m², écart au pont plat, et robustesse.

    Colonne 'si_cap_oppose_%' : le meme bateau parcourant la route en sens
    inverse. Un plan incliné qui gagne dans un sens perd dans l'autre - c'est
    la mesure du pari que l'on prend. Le pont plat, lui, est insensible au cap.
    """
    quotidien = journalier(poa, pas_min)
    quotidien_inv = journalier(poa_inverse, pas_min)

    moy = quotidien.mean()
    moy_inv = quotidien_inv.mean()
    ref, ref_inv = moy["Pont plat (0°)"], moy_inv["Pont plat (0°)"]

    tab = pd.DataFrame(
        {
            "kWh/m2/jour": moy,
            "ecart_vs_plat_%": 100 * (moy / ref - 1),
            "si_cap_oppose_%": 100 * (moy_inv / ref_inv - 1),
            "p10_jour_kWh/m2": quotidien.quantile(0.10),
            "pire_jour_kWh/m2": quotidien.min(),
        }
    )
    # Le critere de decision : la pire des deux orientations de route.
    tab["pire_cas_%"] = tab[["ecart_vs_plat_%", "si_cap_oppose_%"]].min(axis=1)
    tab = tab.sort_values("pire_cas_%", ascending=False)
    return tab, quotidien


def figure(poa, quotidien, tab, route, titre, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"Epi'Tortue - géométrie du pont solaire\n{titre}", fontsize=13)

    # (a) écart au pont plat
    ax = axes[0, 0]
    sous = tab.drop(index=[i for i in tab.index if i.startswith("[réf.]")])
    coul = ["#2a7", "#888"]
    ax.barh(range(len(sous)) , sous["ecart_vs_plat_%"], color="#2a7", height=0.38,
            label="route simulée")
    ax.barh([i + 0.4 for i in range(len(sous))], sous["si_cap_oppose_%"],
            color="#c33", height=0.38, label="cap opposé")
    ax.legend(fontsize=7)
    ax.set_yticks([i + 0.2 for i in range(len(sous))])
    ax.set_yticklabels(sous.index, fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("écart au pont plat (%)")
    ax.set_title("(a) Gain solaire réel selon le sens de la route")
    ax.invert_yaxis()

    # (b) production quotidienne
    ax = axes[0, 1]
    for nom in ["Pont plat (0°)", "Toit 2 pans 25° (latéral)", "Plan incliné 20° vers tribord"]:
        ax.plot(quotidien.index, quotidien[nom], lw=1.2, label=nom)
    ax.set_ylabel("kWh/m²/jour")
    ax.set_title("(b) Production quotidienne le long de la route")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelsize=7, rotation=30)

    # (c) profil moyen d'une journée
    ax = axes[1, 0]
    prof = poa.groupby(poa.index.hour).mean()
    for nom in ["Pont plat (0°)", "Flancs verticaux (4 faces)", "[réf.] suiveur 2 axes"]:
        ax.plot(prof.index, prof[nom], lw=1.4, label=nom)
    ax.set_xlabel("heure UTC")
    ax.set_ylabel("W/m²")
    ax.set_title("(c) Profil journalier moyen")
    ax.legend(fontsize=7)

    # (d) cap réel : la cause de tout
    ax = axes[1, 1]
    ax.plot(route.index, route["cap"], lw=0.6, color="#c33")
    ax.set_ylabel("cap (° / N)")
    ax.set_title("(d) Cap du bateau - imposé par la route, pas par le soleil")
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.set_ylim(0, 360)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(chemin, dpi=130)
    print(f"Figure -> {chemin}")


# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--route", choices=list(ROUTES), default="antilles")
    p.add_argument("--vitesse", type=float, default=2.5, help="noeuds")
    p.add_argument("--depart", default=None, help="AAAA-MM-JJ")
    p.add_argument("--roulis", type=float, default=15.0, help="amplitude roulis (°)")
    p.add_argument("--tangage", type=float, default=6.0, help="amplitude tangage (°)")
    p.add_argument("--lacet", type=float, default=10.0, help="ecart-type du lacet (°)")
    p.add_argument("--pas", type=int, default=30, help="pas de temps (min)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefixe", default="pont_solaire")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    r = ROUTES[args.route]
    depart = args.depart or r["depart"]

    route = construire_route(r["waypoints"], depart, args.vitesse, args.pas, args.lacet, rng)
    sol = ressource_solaire(route, args.pas, rng)
    poa = simuler(route, sol, args.roulis, args.tangage)

    # Meme route, meme meteo, cap inverse : test de robustesse.
    route_inv = route.copy()
    route_inv["cap"] = (route_inv["cap"] + 180.0) % 360.0
    poa_inverse = simuler(route_inv, sol, args.roulis, args.tangage)

    tab, quotidien = synthese(poa, poa_inverse, args.pas)

    jours = (route.index[-1] - route.index[0]).total_seconds() / 86400
    titre = (
        f"{r['nom']} - depart {depart}, {args.vitesse} nd, "
        f"{route['km_parcourus'].iloc[-1]:.0f} km en {jours:.0f} j "
        f"(roulis ±{args.roulis:.0f}°, lacet {args.lacet:.0f}°)"
    )

    print("\n" + "=" * 78)
    print(titre)
    print("=" * 78)
    print(tab.round(3).to_string())
    print("""
Lecture
-------
Energie utile PAR M² DE PANNEAU, a surface installee egale : on compare des
GEOMETRIES, pas des tailles. Ajouter des panneaux sur les flancs ajoute bien
de l'energie en valeur absolue, meme si le rendement par m² y est faible.

  ecart_vs_plat_%  : gain/perte sur la route simulee
  si_cap_oppose_%  : la meme geometrie, route parcourue en sens inverse
  pire_cas_%       : le minimum des deux -> critere de decision, car le cap
                     depend de la route et des deroutes, pas du soleil
  p10_jour         : le 1er decile des journees. C'est lui qui vide la
                     batterie, pas la moyenne.

Les lignes [ref.] sont des bornes inatteignables sur un bateau dont le cap est
impose par la route : elles donnent l'echelle de ce qu'une orientation vers le
soleil rapporterait si elle etait possible.
""")

    reelles = tab.drop(index=[i for i in tab.index if i.startswith("[réf.]")])
    gagnant = reelles["pire_cas_%"].idxmax()
    ecart = reelles.loc[gagnant, "pire_cas_%"]
    print(f"--> Meilleur pire-cas : {gagnant} ({ecart:+.1f} % vs pont plat)")
    incline = reelles.loc[reelles.index.str.startswith("Plan incliné")]
    best = incline["ecart_vs_plat_%"].idxmax()
    print(
        f"--> Meilleur plan incliné sur cette route : {best} "
        f"({incline.loc[best, 'ecart_vs_plat_%']:+.1f} %), mais "
        f"{incline.loc[best, 'si_cap_oppose_%']:+.1f} % si le cap s'inverse.\n"
        "    Le gain est un pari sur le cap ; la perte, elle, est acquise "
        "des que la route change.\n"
    )

    poa.to_csv(f"{args.prefixe}_horaire.csv")
    tab.to_csv(f"{args.prefixe}_synthese.csv")
    figure(poa, quotidien, tab, route, titre, f"{args.prefixe}.png")
    print(f"CSV    -> {args.prefixe}_horaire.csv, {args.prefixe}_synthese.csv")


if __name__ == "__main__":
    main()
