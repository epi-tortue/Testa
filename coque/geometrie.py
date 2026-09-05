"""Primitives géométriques partagées : Bézier rationnelle, lissage (loft) d'anneaux
fermés en maillage étanche. Utilisées par la coque (mesh.py) et les ailes (ailes.py)."""
import numpy as np
import trimesh


def bezier_rationnelle(P0, P1, P2, w, n=80):
    """Courbe passant par P0 et P2, tirée vers P1 avec le poids w (<1 arrondi, >1 anguleux)."""
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    b0 = (1 - t) ** 2
    b1 = 2 * t * (1 - t) * w
    b2 = t ** 2
    return (b0 * P0 + b1 * P1 + b2 * P2) / (b0 + b1 + b2)


def loft(anneaux, centre_sur_axe=False):
    """Peau triangulée entre anneaux successifs + bouchons en éventail aux deux bouts.

    anneaux : array (N, m, 3), N contours fermés de m points chacun, ordonnés dans le
              même sens, sans point doublé.
    Retourne (vertices, faces, j_face) où j_face[f] est l'indice j (0..m-1) de
    l'arête d'anneau dont provient la face f (-1 pour les bouchons) : permet
    d'étiqueter des régions (ex : faces du pont) après coup.
    """
    anneaux = np.asarray(anneaux, dtype=float)
    N, m, _ = anneaux.shape
    vertices = anneaux.reshape(-1, 3)

    def idx(i, j):
        return i * m + (j % m)

    faces, j_face = [], []
    for i in range(N - 1):
        for j in range(m):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            faces.append([a, b, d]); faces.append([a, d, c])
            j_face += [j, j]

    # bouchons orientés comme la peau (arête parcourue en sens inverse de la face voisine)
    # -> enroulement cohérent d'emblée, fix_normals n'a plus qu'à vérifier le signe du volume
    extra = []
    for i_st, sens in ((0, -1), (N - 1, +1)):
        centre = anneaux[i_st].mean(axis=0)
        if centre_sur_axe:
            centre[1] = 0.0
        i_c = len(vertices) + len(extra)
        extra.append(centre)
        for j in range(m):
            a, b = idx(i_st, j), idx(i_st, j + 1)
            faces.append([i_c, a, b] if sens > 0 else [i_c, b, a])
            j_face.append(-1)

    vertices = np.vstack([vertices, np.array(extra)])
    return vertices, np.array(faces), np.array(j_face)


def maillage(vertices, faces):
    """Trimesh fusionné, normales cohérentes et sortantes."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def aire_projetee_z(mesh, mask=None):
    """Aire projetée sur le plan horizontal (somme aire * |n_z|) d'un sous-ensemble de faces."""
    idx = np.arange(len(mesh.faces)) if mask is None else np.flatnonzero(mask)
    return float((mesh.area_faces[idx] * np.abs(mesh.face_normals[idx, 2])).sum())
