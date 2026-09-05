"""Modèle de masses : budget embarqué (params) + structure CALCULÉE sur la géométrie
(coque sandwich, caissons d'ailes, panneaux) + lest interne.

Positions dans le repère du maillage (x = 0 étrave, z = 0 quille au maître-bau) :
composants lourds plaqués au fond (S1), électronique à mi-hauteur, panneaux sur le pont,
structure au centroïde de surface de chaque pièce."""
import numpy as np

from .gz import PointMass, total_mass_and_cg
from .geometrie import aire_projetee_z
from . import params as P


def masse_structure_coque(mesh):
    return mesh.area * P.M_SURF_COQUE


def modele_masses(coque, ailes, LEST=P.DEFAUTS["LEST"], ailes_structure=True):
    """Liste de PointMass pour une CoqueMesh générée et ses Ailes."""
    mesh = coque.mesh
    L, creux = coque.L_COQUE, coque.CREUX
    x_cg = coque.X_MAITRE * L
    z_fond = float(mesh.bounds[0, 2]) + 0.01
    z_pont = float(mesh.bounds[1, 2])
    c_surf = (mesh.triangles_center * mesh.area_faces[:, None]).sum(0) / mesh.area

    ms = [
        PointMass("structure coque", masse_structure_coque(mesh), *c_surf),
        PointMass("batteries", P.M_BATTERIES, x_cg, 0.0, z_fond + 0.035),
        PointMass("moteur+pod", P.M_MOTEUR, 0.88 * L, 0.0, z_fond + 0.03),
        PointMass("electronique", P.M_ELECTRONIQUE, 0.45 * L, 0.0, 0.40 * creux),
        PointMass("divers", P.M_DIVERS, x_cg, 0.0, 0.35 * creux),
    ]
    # panneaux : proportionnels à la surface de pont (coque + ailes), posés sur le pont
    idx = np.flatnonzero(coque.face_pont)
    a_pont = aire_projetee_z(mesh, coque.face_pont)
    x_pont = float((mesh.triangles_center[idx, 0] * mesh.area_faces[idx]).sum() / mesh.area_faces[idx].sum())
    a_ailes = ailes.aire_plan_totale()
    x_ailes = 0.5 * (ailes.x0 + ailes.x1)
    a_tot = a_pont + a_ailes
    x_sol = (x_pont * a_pont + x_ailes * a_ailes) / a_tot
    ms.append(PointMass("panneaux", P.M_SURF_SOLAIRE * a_tot, x_sol, 0.0, z_pont + 0.005))
    if LEST > 0:
        ms.append(PointMass("lest interne", LEST, x_cg, 0.0, z_fond + 0.005))
    if ailes_structure:
        for s in (+1, -1):
            c = ailes.centroide(s)
            ms.append(PointMass(f"aile {'tribord' if s > 0 else 'bâbord'}",
                                ailes.aire_coque(s) * P.M_SURF_AILE, *c))
    return ms


def bilan(masses):
    """Tableau lisible du modèle de masses."""
    M, cg = total_mass_and_cg(masses)
    lignes = [f"{'poste':<18}{'kg':>7}{'x':>8}{'z':>8}"]
    for m in masses:
        lignes.append(f"{m.name:<18}{m.mass_kg:7.2f}{m.x:8.3f}{m.z:8.3f}")
    lignes.append(f"{'TOTAL':<18}{M:7.2f}{cg[0]:8.3f}{cg[2]:8.3f}")
    return "\n".join(lignes)
