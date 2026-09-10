"""Sorties : STL (coque, ailes, ensemble), JSON des valeurs, tracé de la courbe GZ."""
import json
import math
import os

import numpy as np
import trimesh

from . import params as P


def exporter(res, dossier="outputs", prefixe="coque"):
    """Écrit <prefixe>.stl (coque seule), <prefixe>_ailes.stl, <prefixe>_ensemble.stl,
    <prefixe>.json (valeurs + résumé chiffré) et <prefixe>_gz.png si la courbe existe.
    Retourne la liste des fichiers écrits."""
    os.makedirs(dossier, exist_ok=True)
    ecrits = []
    if "coque" in res:
        coque, ailes = res["coque"], res["ailes"]
        p = os.path.join(dossier, f"{prefixe}.stl")
        coque.mesh.export(p); ecrits.append(p)
        wings = trimesh.util.concatenate([ailes.meshes[+1], ailes.meshes[-1]])
        p = os.path.join(dossier, f"{prefixe}_ailes.stl")
        wings.export(p); ecrits.append(p)
        p = os.path.join(dossier, f"{prefixe}_ensemble.stl")
        trimesh.util.concatenate([coque.mesh, wings]).export(p); ecrits.append(p)
    p = os.path.join(dossier, f"{prefixe}.json")
    with open(p, "w") as fh:
        json.dump(resume_json(res), fh, indent=1, ensure_ascii=False)
    ecrits.append(p)
    if res.get("valide") and "phi" in res["stab"]:
        p = os.path.join(dossier, f"{prefixe}_gz.png")
        tracer_gz({prefixe: res}, p); ecrits.append(p)
    return ecrits


def _f(x):
    """float sérialisable (None si absent ou non fini : JSON n'a pas d'Infinity)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def resume_json(res):
    """Sous-ensemble sérialisable du résultat d'objectif.evaluer."""
    out = dict(valeurs={k: float(v) for k, v in res["valeurs"].items()},
               valide=bool(res["valide"]), score=_f(res["score"]), raison=res.get("raison", ""))
    if "largeur_hors_tout" in res:
        out.update(largeur_hors_tout=_f(res["largeur_hors_tout"]), volume_ailes_L=_f(res["volume_ailes_L"]))
    if "masses" in res:
        out["masses"] = [dict(nom=m.name, kg=_f(m.mass_kg), x=_f(m.x), y=_f(m.y), z=_f(m.z)) for m in res["masses"]]
    if res.get("valide"):
        st, h = res["stab"], res["hydro"]
        out["hydro"] = {k: _f(h[k]) for k in ("T", "volume", "S", "Lwl", "Bwl", "Cb", "Cp")}
        out["energie"] = dict(surface_panneaux=_f(res["surface_panneaux"]), production_Wh_j=_f(res["production_Wh_j"]),
                              E_Wh_km=_f(res["E_Wh_km"]), score_prop=_f(res["score_prop"]), penalite=_f(res["penalite"]))
        cles = ("M", "KG", "LCG", "T", "franc_bord", "garde_ailes", "GZ_172", "GZ_min_B", "phi_GZmin",
                "GZ_max", "phi_GZmax", "AVS", "GM0", "GM0_grille", "dGZ180", "aire_pos", "centre_aire",
                "GZ_min_tot", "aire_60", "gite_vent_moyen", "gite_vent_fort", "periode_roulis",
                "profondeur_trou_min",
                "phi_inondation_bas", "GM0_A", "GZ_max_A", "AVS_A")
        out["stabilite"] = dict(go=bool(st["go"]), raison=st.get("raison", ""),
                                **{k: _f(st[k]) for k in cles if k in st})
        if "phi" in st:
            out["stabilite"]["phi"] = [float(x) for x in st["phi"]]
            out["stabilite"]["GZ"] = [float(x) for x in st["GZ"]]
            out["stabilite"]["GZ_rob"] = [float(x) for x in st["GZ_rob"]]
            out["stabilite"]["phi_inondation"] = {str(k): v for k, v in st["phi_inondation"].items()}
            out["stabilite"]["pieges_ailes_seches"] = [{k: _f(x) for k, x in p.items()} for p in st.get("pieges", [])]
            for k in ("GZ_A", "GZ_B"):
                if k in st:
                    out["stabilite"][k] = [float(x) for x in st[k]]
    return out


def tracer_gz(cas, chemin, titre=None):
    """cas : {label: res} (résultats d'objectif.evaluer avec courbe). Trace la courbe réelle en
    trait plein, et si présentes les courbes A (pointillé) et B (tireté)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (lab, res) in enumerate(cas.items()):
        st = res["stab"]
        if "phi" not in st:
            continue
        col = f"C{i}"
        ax.plot(st["phi"], st["GZ"] * 100, color=col, lw=2,
                label=f"{lab} - réel (GO={st['go']}, GZmin[90,170]={st['GZ_min_B']*100:+.1f} cm)")
        if "GZ_A" in st:
            ax.plot(st["phi"], st["GZ_A"] * 100, color=col, lw=1, ls=":", label=f"{lab} - A ailes sèches")
        if "GZ_B" in st:
            ax.plot(st["phi"], st["GZ_B"] * 100, color=col, lw=1, ls="--", label=f"{lab} - B ailes noyées")
        for s, mk in ((+1, "v"), (-1, "^")):
            a = st["phi_inondation"].get(s)
            if a is not None:
                ax.plot([a], [np.interp(a, st["phi"], st["GZ"]) * 100], mk, color=col)
    ax.axhline(0, color="k", lw=0.8)
    lo, hi = ax.get_ylim()
    lo, hi = min(-3.0, lo), max(3.0, hi)
    ax.axhspan(lo, P.MARGE_GZ_MIN * 100, xmin=P.PLAGE_GZ_MIN[0] / 180, xmax=P.PLAGE_GZ_MIN[1] / 180,
               color="red", alpha=0.06, label=f"zone interdite : GZ < {P.MARGE_GZ_MIN*100:.0f} cm sur [90°,170°]")
    ax.set_xlim(0, 180); ax.set_ylim(lo, hi)
    ax.set_xlabel("Gîte φ (°)"); ax.set_ylabel("GZ (cm)")
    ax.set_title(titre or "Bras de levier de redressement (▼ inondation aile tribord, ▲ bâbord)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(chemin, dpi=150); plt.close(fig)
    return chemin
