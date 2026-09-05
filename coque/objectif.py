"""Évaluation complète d'un candidat -> score scalaire pour l'optimiseur.

    score = production solaire journalière [Wh/j] / énergie pour 1 km à V_CROISIERE [Wh/km]
            - pénalités (auto-redressement, franc-bord, ailes dans l'eau, largeur hors-tout)

Le score s'interprète en "km/jour d'énergie" : plus il est grand, plus la plateforme
produit par rapport à ce qu'elle consomme. Les contraintes DURES (géométrie impossible,
maillage non étanche, coque qui coule) reçoivent SCORE_INVALIDE, strictement pire que
n'importe quelle coque valide même très pénalisée : l'optimiseur ne peut pas se réfugier
dans la zone invalide.
"""
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

    stab = verdict_auto_redressement(coque, ailes, masses, pas=pas, assiette_libre=assiette_libre,
                                     courbes_AB=courbes_AB)
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
    prod = production_journaliere(coque, ailes)
    s = prod / e["E_Wh"]
    pen = penalite(stab) + K_PENALITE * max(0.0, res["largeur_hors_tout"] - P.B_HORS_TOUT_MAX)
    res.update(valide=True, hydro=h, prop=e,
               surface_panneaux=surface_panneaux(coque, ailes), production_Wh_j=prod,
               E_Wh_km=e["E_Wh"], score_prop=s, penalite=pen, score=s - pen)
    return res


def score(curseur, verbose=False):
    """Point d'entrée de l'optimiseur : curseur normalisé -> score (à MAXIMISER)."""
    res = evaluer(P.curseur_vers_valeurs(curseur))
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
            f"GM0={st.get('GM0', float('nan'))*100:.1f} cm | M={res['M']:.1f} kg KG={res['KG']*100:.1f} cm "
            f"T={res['hydro']['T']*100:.1f} cm fb={st.get('franc_bord', float('nan'))*100:.1f} cm | "
            f"B_tot={res['largeur_hors_tout']:.2f} m S_pan={res['surface_panneaux']:.2f} m² "
            f"{res['production_Wh_j']:.0f} Wh/j {res['E_Wh_km']:.2f} Wh/km  {st.get('raison', '')}")


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
                   "=== Stabilité (courbe réelle, ailes noyées à l'immersion des trous) ===",
                   f"GO={st['go']}  {st['raison']}"]
        if "phi" in st:
            lignes += [f"GZ_max {g('GZ_max'):.1f} cm @ {st['phi_GZmax']:.0f}°  GZ_min[90,170] "
                       f"{g('GZ_min_B'):+.1f} cm @ {st['phi_GZmin']:.0f}°  AVS {st['AVS']:.0f}°  "
                       f"GM0 {g('GM0'):.1f} cm  dGZ/dφ(180°) {g('dGZ180'):+.1f} cm/rad",
                       f"inondation des ailes : tribord {st['phi_inondation'][+1]}°, bâbord {st['phi_inondation'][-1]}°"]
        elif "GZ_172" in st:
            lignes.append(f"GZ(172°) = {g('GZ_172'):+.1f} cm (filtre rapide)")
        lignes += ["=== Énergie ===",
                   f"panneaux {res['surface_panneaux']:.2f} m²  production {res['production_Wh_j']:.0f} Wh/j  "
                   f"à {P.V_CROISIERE} m/s : {e['Rt']:.2f} N (air {e['R_air']:.2f} N)  "
                   f"{e['P_absorbee']:.1f} W  {res['E_Wh_km']:.2f} Wh/km",
                   f"SCORE {res['score']:.2f}  (prop {res['score_prop']:.2f}, pénalité {res['penalite']:.2f})"]
    else:
        lignes.append(f"REJETÉ : {res['raison']}")
    return "\n".join(lignes)
