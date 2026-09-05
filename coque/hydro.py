"""Hydrostatique droite : tirant d'eau d'équilibre et géométrie de la carène mouillée.

    masse -> tirant d'eau (équilibre vertical, assiette nulle)
    carène immergée -> volume, surface mouillée, Lwl, Bwl, Cb, Cp

Repère du maillage : x longitudinal, y transversal, z vertical vers le haut.
La stabilité aux grands angles (carènes inclinées) est dans gz.py.
"""
import numpy as np
import trimesh
from trimesh import intersections

from .params import RHO_EAU


class Carene:
    """Équilibre vertical d'un maillage étanche + géométrie de la partie immergée."""

    def __init__(self, mesh, masse, rho=RHO_EAU):
        self.mesh = mesh
        self.masse = masse          # [kg] déplacement total
        self.rho = rho
        self._res = None

    def _immersion(self, z):
        """Coupe au plan z : (triangles immergés, volume sous le plan).

        Volume par le théorème de la divergence avec F = (0, 0, z - z0) : le couvercle
        plan contribue pour zéro, inutile de refermer le maillage.
        """
        out = intersections.slice_faces_plane(
            self.mesh.vertices, self.mesh.faces,
            plane_normal=np.array([0.0, 0.0, -1.0]),
            plane_origin=np.array([0.0, 0.0, z]))
        v, f = out[0], out[1]
        if len(f) == 0:
            return None, 0.0
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        zc = tri[:, :, 2].mean(axis=1)
        return tri, abs(float(np.sum(0.5 * n[:, 2] * (zc - z))))

    def volume_sous(self, z):
        """Volume immergé [m3] si la flottaison est à la cote z."""
        return self._immersion(z)[1]

    def resoudre(self, tol=1e-5):
        """Dichotomie sur z tel que rho * V(z) = masse. None si la coque coule."""
        if self._res is not None:
            return self._res
        v_cible = self.masse / self.rho
        z_min, z_max = self.mesh.bounds[0, 2], self.mesh.bounds[1, 2]
        if self.volume_sous(z_max) < v_cible:
            return None
        lo, hi = z_min, z_max
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.volume_sous(mid) < v_cible:
                lo = mid
            else:
                hi = mid
        z = 0.5 * (lo + hi)
        tri, vol = self._immersion(z)

        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        S = float(0.5 * np.linalg.norm(n, axis=1).sum())      # surface mouillée
        pts = tri.reshape(-1, 3)
        flot = pts[pts[:, 2] > z - 1e-6]                        # trace de la flottaison
        Lwl = float(flot[:, 0].max() - flot[:, 0].min())
        Bwl = float(2.0 * np.abs(flot[:, 1]).max())
        T = float(z - z_min)

        # aire de la section immergée maximale (pour Cp) : coupes transversales
        A_max = 0.0
        for x in np.linspace(flot[:, 0].min() + 1e-3, flot[:, 0].max() - 1e-3, 21):
            A_max = max(A_max, self._aire_section(tri, x))

        self._res = dict(
            z_flottaison=z, T=T, volume=vol, S=S, Lwl=Lwl, Bwl=Bwl,
            Cb=vol / (Lwl * Bwl * T),
            Cp=vol / (Lwl * A_max) if A_max > 0 else float("nan"),
            A_maitre=A_max,
            x_tableau=float(self.mesh.bounds[1, 0]),
        )
        return self._res

    @staticmethod
    def _aire_section(tri, x):
        """Aire de la section transversale immergée au droit de x. La section est symétrique
        et sa frontière tribord y(z) est univoque : A = 2 * ∫ y_tribord(z) dz."""
        m = trimesh.Trimesh(vertices=tri.reshape(-1, 3),
                            faces=np.arange(len(tri) * 3).reshape(-1, 3), process=False)
        seg = intersections.mesh_plane(m, plane_normal=[1.0, 0.0, 0.0], plane_origin=[x, 0.0, 0.0])
        if len(seg) == 0:
            return 0.0
        pts = seg.reshape(-1, 3)
        pts = pts[pts[:, 1] >= -1e-9]
        if len(pts) < 2:
            return 0.0
        order = np.argsort(pts[:, 2])
        z, y = pts[order, 2], pts[order, 1]
        return 2.0 * float(np.trapezoid(y, z))
