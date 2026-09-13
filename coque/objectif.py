"""Évaluation complète d'un candidat -> score scalaire pour l'optimiseur.

    score = production solaire journalière [Wh/j] × cos(gîte moyenne sous VENT_MOYEN)
            / énergie pour 1 km à V_CROISIERE [Wh/km]
            - pénalités (auto-redressement robuste à KG + 1 cm, tenue au vent, réserve
              dynamique, franc-bord, ailes dans l'eau, largeur hors-tout)

Le score s'interprète en "km/jour d'énergie" : plus il est grand, plus la plateforme
produit par rapport à ce qu'elle consomme. Les contraintes DURES (géométrie impossible,
maillage non étanche, coque qui coule) reçoivent SCORE_INVALIDE, strictement pire que
n'importe quelle coque valide même très pénalisée : l'optimiseur ne peut pas se réfugier
dans la zone invalide.
"""
import math

from .mesh import CoqueMesh
from .ailes import Ailes
from .masses import modele_masses
from .stabilite import verdict_auto_redressement, penalite
from .propulsion import Propulsion
from .energie import production_journaliere, surface_panneaux
from . import params as P

SCORE_INVALIDE = -1.0e4
K_PENALITE = 20000.0          # [points/m] même échelle que stabilite.penalite


def evaluer(valeurs, pas=P.PAS_GZ_OPTIM, assiette_libre=False, courbes_AB=False):
    """Dict complet : géométrie, masses, stabilité, hydro, énergie, score."""
    res = dict(valeurs=dict(valeurs), valide=False, score=SCORE_INVALIDE, raison="")
    ok, msg = P.candidat_valide(valeurs)
    if not ok:
        res["raison"] = f"candidat invalide : {msg}"
        return res
    kc, ka, km = P.separer(valeurs)

    coque = CoqueMesh(**kc)
    mesh = coque.generate()
    if not mesh.is_watertight:
        res["raison"] = "maillage non étanche"
        return res
    ailes = Ailes.depuis_coque(coque, **ka)
    masses = modele_masses(coque, ailes, **km)
    res.update(coque=coque, ailes=ailes, masses=masses,
               largeur_hors_tout=P.largeur_hors_tout(valeurs),
               volume_ailes_L=ailes.volume_total() * 1000.0)

    grossier = pas > P.PAS_GZ_RAPPORT
    stab = verdict_auto_redressement(coque, ailes, masses, pas=pas, assiette_libre=assiette_libre,
                                     courbes_AB=courbes_AB,
                                     marge=P.MARGE_GRILLE if grossier else 0.0,
                                     marge_deg=P.MARGE_GRILLE_DEG if grossier else 0.0)
    res["stab"] = stab
    res["raison"] = stab.get("raison", "")
    if "M" not in stab:
        return res
    res.update(M=stab["M"], KG=stab["KG"], LCG=stab["LCG"])

    prop = Propulsion(mesh, masse=stab["M"], x_cg=stab["LCG"], deadrise_deg=kc["DEADRISE"],
                      vent=P.VENT_APPARENT, aire_frontale_extra=ailes.aire_frontale())
    h = prop.carene.resoudre()
    if h is None:
        res["raison"] = "coule (hydro)"
        return res
    e = prop.energie(P.V_CROISIERE, 1000.0)
    facteur_gite = math.cos(math.radians(min(stab.get("gite_vent_moyen", 0.0), 90.0)))
    prod = production_journaliere(coque, ailes) * facteur_gite
    s = prod / e["E_Wh"]
    pen_largeur = K_PENALITE * max(0.0, res["largeur_hors_tout"] - P.B_HORS_TOUT_MAX)
    pen = penalite(stab) + pen_largeur
    if pen_largeur > 0 and stab.get("go"):
        # le verdict de stabilité est bon mais la plateforme est trop large : NO-GO global
        stab["go"] = False
        stab["raison"] = (f"largeur hors-tout {res['largeur_hors_tout']:.3f} m > {P.B_HORS_TOUT_MAX:.2f} m"
                          + (" ; " + stab["raison"] if stab.get("raison") else ""))
    res.update(valide=True, hydro=h, prop=e, facteur_gite=facteur_gite,
               surface_panneaux=surface_panneaux(coque, ailes), production_Wh_j=prod,
               E_Wh_km=e["E_Wh"], score_prop=s, penalite=pen, score=s - pen)
    return res


def score(curseur, verbose=False, pas=P.PAS_GZ_OPTIM, assiette_libre=False):
    """Point d'entrée de l'optimiseur : curseur normalisé -> score (à MAXIMISER)."""
    res = evaluer(P.curseur_vers_valeurs(curseur), pas=pas, assiette_libre=assiette_libre)
    if verbose:
        print(resume(res))
    return res["score"]


def resume(res):
    """Une ligne lisible."""
    if not res["valide"]:
        return f"score {res['score']:9.1f}  {res['raison']}"
    st = res["stab"]
    return (f"score {res['score']:8.2f} = prop {res['score_prop']:7.2f} - pén {res['penalite']:7.2f} | "
            f"GO={st['go']} GZmin[90,170]={st.get('GZ_min_B', float('nan'))*100:+.1f} cm "
            f"GM0={st.get('GM0', float('nan'))*100:.1f} cm inond={st.get('phi_inondation_bas', float('nan')):.0f}° "
            f"gîte({P.VENT_GITE:.0f}m/s)={st.get('gite_vent_fort', float('nan')):.1f}° "
            f"aire60={st.get('aire_60', float('nan'))*1000:.1f} | "
            f"M={res['M']:.1f} kg KG={res['KG']*100:.1f} cm "
            f"T={res['hydro']['T']*100:.1f} cm fb={st.get('franc_bord', float('nan'))*100:.1f} cm | "
            f"B_tot={res['largeur_hors_tout']:.2f} m S_pan={res['surface_panneaux']:.2f} m² "
            f"{res['production_Wh_j']:.0f} Wh/j {res['E_Wh_km']:.2f} Wh/km  {st.get('raison', '')}")


def tableau_parametres(valeurs):
    """Variables de conception avec leurs bornes ; les valeurs à moins de 2 % d'une borne
    sont marquées (l'optimiseur y est contraint par la borne, pas par la physique)."""
    lignes = [f"{'variable':<16}{'valeur':>10}{'min':>9}{'max':>9}{'position':>10}"]
    n_butee = 0
    for k, (lo, hi) in P.VARIABLES_LIBRES.items():
        v = valeurs[k]
        pos = 0.5 if hi == lo else (v - lo) / (hi - lo)
        marque = ""
        if pos <= 0.02:
            marque, n_butee = "  <- butée basse", n_butee + 1
        elif pos >= 0.98:
            marque, n_butee = "  <- butée HAUTE", n_butee + 1
        lignes.append(f"{k:<16}{v:10.4g}{lo:9.4g}{hi:9.4g}{pos*100:9.0f}%{marque}")
    lignes.append(f"{n_butee} variable(s) en butée sur {len(P.VARIABLES_LIBRES)}")
    return "\n".join(lignes)


def bilan_marges(res):
    """Chaque contrainte : valeur, seuil, marge (positive = tenue). Les seuils incluent
    les surcotes de grille (marge, marge_deg) de l'évaluation."""
    st = res.get("stab", {})
    if "phi" not in st:
        return f"contraintes non évaluées : {st.get('raison', res.get('raison', ''))}"
    m, md = st.get("marge", 0.0), st.get("marge_deg", 0.0)
    cm = 100.0
    prof = st["profondeur_trou_min"]
    rows = [  # (nom, valeur, seuil, sens, unité)  sens '>=' ou '<='
        ("GZ_min [90°,170°]  (KG+1cm)", st["GZ_min_B"] * cm, (P.MARGE_GZ_MIN + m) * cm, ">=", "cm"),
        ("GZ_min ]0°,180°[   (KG+1cm)", st["GZ_min_tot"] * cm, m * cm, ">=", "cm"),
        ("GM0 à 2°           (KG+1cm)", st["GM0"] * cm, (P.GM0_MIN + m) * cm, ">=", "cm"),
        ("aire GZ 0-60°", st["aire_60"] * 1000, (P.AIRE_GZ_MIN + m * math.radians(P.PHI_AIRE)) * 1000, ">=", "mm.rad"),
        ("inondation aile basse", st["phi_inondation_bas"], P.PHI_INONDATION_MIN + md, ">=", "°"),
        (f"gîte sous {P.VENT_GITE:.0f} m/s", st["gite_vent_fort"], P.GITE_VENT_MAX - md, "<=", "°"),
        (f"inondation - gîte({P.VENT_GITE:.0f} m/s)", st["phi_inondation_bas"] - st["gite_vent_fort"],
         P.MARGE_INOND_VENT + md, ">=", "°"),
        (f"gîte sous {P.VENT_MOYEN:.0f} m/s (panneaux)", st["gite_vent_moyen"], P.GITE_MOYENNE_MAX - md, "<=", "°"),
        (f"contact aile - gîte({P.VENT_MOYEN:.0f} m/s)", st["angle_contact_aile"] - st["gite_vent_moyen"],
         P.MARGE_CONTACT_AILE + md, ">=", "°"),
        ("trous au piège ailes sèches", (prof * cm if math.isfinite(prof) else float("inf")),
         (P.PROFONDEUR_TROU_MIN + m) * cm, ">=", "cm"),
        ("franc-bord", st["franc_bord"] * cm, P.FRANC_BORD_MINI * cm, ">=", "cm"),
        ("garde des ailes au repos", st["garde_ailes"] * cm, 0.0, ">=", "cm"),
        ("largeur hors-tout", res["largeur_hors_tout"], P.B_HORS_TOUT_MAX, "<=", "m"),
    ]
    lignes = [f"{'contrainte':<34}{'valeur':>10}   {'seuil':>10}{'marge':>10}"]
    for nom, val, seuil, sens, u in rows:
        marge = (val - seuil) if sens == ">=" else (seuil - val)
        if not math.isfinite(val):
            lignes.append(f"{nom:<34}{'aucun':>10}   {sens} {seuil:8.2f}{'':>10} {u}")
            continue
        flag = "  <- VIOLÉE" if marge < -1e-9 else ("  <- au seuil" if marge < 0.15 * max(abs(seuil), 1.0) + 1e-9 else "")
        lignes.append(f"{nom:<34}{val:10.2f}   {sens} {seuil:8.2f}{marge:+10.2f} {u}{flag}")
    return "\n".join(lignes)


def rapport(res):
    """Rapport multi-lignes (masses, hydro, stabilité, énergie)."""
    from .masses import bilan
    v = res["valeurs"]
    lignes = ["=== Candidat ===",
              ", ".join(f"{k}={v[k]:.4g}" for k in P.VARIABLES_LIBRES),
              f"largeur hors-tout {res.get('largeur_hors_tout', float('nan')):.3f} m  "
              f"volume ailes {res.get('volume_ailes_L', float('nan')):.1f} L"]
    if "masses" in res:
        lignes += ["=== Masses ===", bilan(res["masses"])]
    if res["valide"]:
        h, st, e = res["hydro"], res["stab"], res["prop"]
        g = lambda k, f=100.0: (st[k] * f) if k in st else float("nan")
        lignes += ["=== Hydrostatique au repos ===",
                   f"T {h['T']*100:.1f} cm  volume {h['volume']*1000:.1f} L  S mouillée {h['S']:.3f} m²  "
                   f"Lwl {h['Lwl']:.2f} m  Bwl {h['Bwl']:.3f} m  Cb {h['Cb']:.3f}  Cp {h['Cp']:.3f}",
                   f"franc-bord {g('franc_bord'):.1f} cm  garde ailes {g('garde_ailes'):.1f} cm",
                   f"=== Stabilité (courbe réelle, ailes noyées à l'immersion des trous ; "
                   f"critères à KG + {P.MARGE_KG*100:.0f} cm) ===",
                   f"GO={st['go']}  {st['raison']}"]
        if "phi" in st:
            lignes += [f"GZ_max {g('GZ_max'):.1f} cm @ {st['phi_GZmax']:.0f}°  GZ_min[90,170] "
                       f"{g('GZ_min_B'):+.1f} cm @ {st['phi_GZmin']:.0f}°  AVS {st['AVS']:.0f}°  "
                       f"GM0 {g('GM0'):.1f} cm (à {P.PHI_GM0:.0f}°)  dGZ/dφ(180°) {g('dGZ180'):+.1f} cm/rad",
                       f"inondation des ailes : tribord {st['phi_inondation'][+1]}°, bâbord {st['phi_inondation'][-1]}°"
                       f"  (trous émergés jusqu'à {st['phi_inondation_bas']:.2f}°, mini {P.PHI_INONDATION_MIN:.0f}°)"]
            ga = st.get("GZ_apres_inondation", {})
            if ga.get(+1) is not None:
                lignes.append(f"GZ juste après inondation tribord {ga[+1]*100:+.1f} cm"
                              f"  ->  GZ_min sur ]0,180[ (grille + post-inondation) {g('GZ_min_tot'):+.1f} cm")
            lignes += [f"tenue au vent : gîte {st['gite_vent_moyen']:.1f}° sous {P.VENT_MOYEN:.0f} m/s "
                       f"(production × {res['facteur_gite']:.3f}, aile basse touche l'eau à "
                       f"{st['angle_contact_aile']:.1f}°), {st['gite_vent_fort']:.1f}° sous "
                       f"{P.VENT_GITE:.0f} m/s (maxi {P.GITE_VENT_MAX:.0f}°, aile basse inondée à "
                       f"{st['phi_inondation_bas']:.1f}°)",
                       f"réserve dynamique : aire GZ 0-{P.PHI_AIRE:.0f}° {st['aire_60']*1000:.1f} mm.rad "
                       f"(mini {P.AIRE_GZ_MIN*1000:.0f})  période de roulis {st['periode_roulis']:.1f} s"]
            if st["pieges"]:
                lignes.append("ailes sèches après chavirage : équilibre(s) stable(s) à "
                              + ", ".join(f"{p['phi']:.0f}° (trous à {p['profondeur_trou']*100:+.1f} cm sous l'eau)"
                                          for p in st["pieges"])
                              + f"  (mini {P.PROFONDEUR_TROU_MIN*100:.0f} cm pour que l'aile se remplisse)")
            else:
                lignes.append("ailes sèches après chavirage : aucun équilibre stable entre 90° et 180°")
        elif "GZ_172" in st:
            lignes.append(f"GZ(172°) = {g('GZ_172'):+.1f} cm (filtre rapide)")
        lignes += ["=== Énergie ===",
                   f"panneaux {res['surface_panneaux']:.2f} m²  production {res['production_Wh_j']:.0f} Wh/j  "
                   f"à {P.V_CROISIERE} m/s : {e['Rt']:.2f} N (air {e['R_air']:.2f} N)  "
                   f"{e['P_absorbee']:.1f} W  {res['E_Wh_km']:.2f} Wh/km",
                   f"SCORE {res['score']:.2f}  (prop {res['score_prop']:.2f}, pénalité {res['penalite']:.2f})",
                   "=== Marges sur les contraintes ===", bilan_marges(res)]
    else:
        lignes.append(f"REJETÉ : {res['raison']}")
    lignes += ["=== Variables de conception et bornes ===", tableau_parametres(v)]
    return "\n".join(lignes)
