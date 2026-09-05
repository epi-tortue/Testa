"""Courbe de bras de levier GZ(phi) 0-180° par la méthode des carènes inclinées.

Repère : x longitudinal (0 = étrave), y transversal (tribord > 0), z vertical vers le haut.

Pour chaque gîte phi, le bateau est incliné (rotation de -phi autour de x : tribord vers le
bas pour phi > 0) puis on cherche le plan de flottaison (hauteur zw, assiette theta) tel que
    (1) rho * V_immergé = masse totale          (équilibre vertical)
    (2) x_B = x_G                                (équilibre longitudinal, si free_trim)
Le volume immergé est la somme des morceaux de chaque corps flottant ACTIF coupés par le plan
(trimesh.slice_mesh_plane, cap=True). GZ = y_B - y_G dans le repère incliné ; GZ > 0 ramène
vers la position droite.

Les corps flottants sont indexés : 0 = coque, puis les ailes. Une aile inondée n'est plus
"active" (l'eau intérieure a la densité de l'eau extérieure : poussée nette nulle) ; seule sa
masse de structure compte, elle est dans la liste des masses ponctuelles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import trimesh
from scipy.optimize import brentq, fsolve

from .params import RHO_EAU, G


@dataclass
class PointMass:
    name: str
    mass_kg: float
    x: float
    y: float
    z: float


def total_mass_and_cg(masses: Iterable[PointMass]) -> tuple[float, np.ndarray]:
    ms = np.array([m.mass_kg for m in masses])
    xyz = np.array([[m.x, m.y, m.z] for m in masses])
    M = ms.sum()
    return float(M), (ms[:, None] * xyz).sum(axis=0) / M


def roll_inertia(masses: Iterable[PointMass], cg: np.ndarray) -> float:
    """Inertie de roulis autour de l'axe x passant par G [kg.m2]."""
    return sum(m.mass_kg * ((m.y - cg[1]) ** 2 + (m.z - cg[2]) ** 2) for m in masses)


class Vessel:
    """Corps flottants + masses ponctuelles. `actifs` = indices des corps qui portent."""

    def __init__(self, bodies: list[trimesh.Trimesh], masses: list[PointMass],
                 rho: float = RHO_EAU, label: str = ""):
        self.bodies = bodies
        self.masses = list(masses)
        self.rho = rho
        self.label = label
        self.M, self.cg = total_mass_and_cg(self.masses)

    def flotte(self, actifs=None) -> bool:
        actifs = range(len(self.bodies)) if actifs is None else actifs
        return self.M <= self.rho * sum(abs(self.bodies[i].volume) for i in actifs)

    # -- volume immergé pour un plan donné, dans le repère incliné
    @staticmethod
    def _immersed(bodies_rot, zw, theta):
        """(volume, centroïde) sous le plan passant par (0,0,zw), de normale
        (sin theta, 0, cos theta) ; l'eau est du côté -normale."""
        n = np.array([math.sin(theta), 0.0, math.cos(theta)])
        V, C = 0.0, np.zeros(3)
        for b in bodies_rot:
            try:
                s = trimesh.intersections.slice_mesh_plane(b, plane_normal=-n,
                                                           plane_origin=[0, 0, zw], cap=True)
            except Exception:
                continue
            if s is None or len(s.faces) == 0:
                continue
            v = abs(s.volume)
            if v < 1e-12:
                continue
            V += v
            C += v * s.center_mass
        return V, (C / V if V > 0 else C)

    def equilibrium(self, phi_deg: float, free_trim: bool = True, actifs=None) -> dict:
        """Équilibre à la gîte phi (deg). Retourne GZ, V, B, G, zw, trim, et la rotation R
        pour tester l'immersion d'un point du bateau (voir `immerge`)."""
        actifs = list(range(len(self.bodies))) if actifs is None else list(actifs)
        R = trimesh.transformations.rotation_matrix(math.radians(-phi_deg), [1, 0, 0], [0, 0, 0])
        bodies_rot = [self.bodies[i].copy().apply_transform(R) for i in actifs]
        g_rot = R[:3, :3] @ self.cg
        target = self.M / self.rho
        zmin = min(b.bounds[0, 2] for b in bodies_rot)
        zmax = max(b.bounds[1, 2] for b in bodies_rot)

        def vol_err(zw, theta=0.0):
            return self._immersed(bodies_rot, zw, theta)[0] - target

        if vol_err(zmax + 1e-4) < 0:
            raise ValueError(f"[{self.label}] {self.M:.1f} kg > flottabilité des corps actifs : coule")
        zw = brentq(vol_err, zmin - 1e-4, zmax + 1e-4, xtol=1e-6)
        theta = 0.0
        if free_trim:
            def resid(p):
                V, C = self._immersed(bodies_rot, p[0], p[1])
                return [(V - target) / target, (C[0] - g_rot[0])]
            sol, info, ier, _ = fsolve(resid, [zw, 0.0], full_output=True, xtol=1e-7)
            if ier == 1 and abs(sol[1]) < math.radians(25):
                zw, theta = float(sol[0]), float(sol[1])
        V, B = self._immersed(bodies_rot, zw, theta)
        return dict(phi=phi_deg, GZ=float(B[1] - g_rot[1]), V=V, B=B.tolist(), G=g_rot.tolist(),
                    zw=zw, theta=theta, trim_deg=math.degrees(theta), R=R, actifs=actifs)

    @staticmethod
    def immerge(point, eq) -> bool:
        """Le point (repère bateau) est-il sous le plan de flottaison de l'équilibre `eq` ?"""
        p = eq["R"][:3, :3] @ np.asarray(point, dtype=float)
        n = np.array([math.sin(eq["theta"]), 0.0, math.cos(eq["theta"])])
        return bool(n @ (p - np.array([0.0, 0.0, eq["zw"]])) < 0.0)

    def gz_curve(self, angles_deg, free_trim: bool = True, actifs=None) -> dict:
        rows = [self.equilibrium(a, free_trim, actifs) for a in angles_deg]
        return dict(label=self.label, phi=np.array([r["phi"] for r in rows]),
                    GZ=np.array([r["GZ"] for r in rows]), rows=rows, M=self.M, cg=self.cg.tolist())


def metrics(phi: np.ndarray, gz: np.ndarray, M: float, tol: float = 1e-4) -> dict:
    """Métriques d'une courbe GZ sur [0,180] :
    go (GZ >= 0 sur ]0,180[), GZ_max, AVS, GZ_min intérieur, GM0 (pente à l'origine),
    GM_inverted (pente en 180° : > 0 = position retournée STABLE = échec), aire positive
    (énergie de redressement / Mg) et son centre, aire négative (piège de la position retournée)."""
    phi_r = np.radians(phi)
    inner = (phi > 0.5) & (phi < 179.5)
    gz_in, ph_in = gz[inner], phi[inner]
    i_max, i_min = int(np.argmax(gz)), int(np.argmin(gz_in))
    avs = 180.0
    for k in range(i_max, len(phi) - 1):
        if gz[k] > 0 >= gz[k + 1]:
            avs = float(phi[k] + (phi[k + 1] - phi[k]) * gz[k] / (gz[k] - gz[k + 1]))
            break
    gm0 = (gz[1] - gz[0]) / (phi_r[1] - phi_r[0])
    gm180 = (gz[-1] - gz[-2]) / (phi_r[-1] - phi_r[-2])
    pos = np.clip(gz, 0, None)
    area = float(np.trapezoid(pos, phi_r))
    centroid = float(np.trapezoid(pos * phi, phi_r) / area) if area > 0 else float("nan")
    neg_area = float(-np.trapezoid(np.clip(gz, None, 0), phi_r))
    return dict(
        go=bool(gz_in.min() > -tol), GZ_max=float(gz[i_max]), phi_GZmax=float(phi[i_max]),
        AVS_deg=avs, GZ_min_inner=float(gz_in[i_min]), phi_GZmin=float(ph_in[i_min]),
        GM0_m=float(gm0), GM_inverted_m=float(gm180),
        area_m_rad=area, area_centroid_deg=centroid, neg_area_m_rad=neg_area,
        righting_energy_J=area * M * G, inverted_trap_energy_J=neg_area * M * G,
    )


def righting_time(phi, gz, M, I_roll, phi0_deg=175.0, added_mass_factor=1.5,
                  damping_ratio=0.0, t_max=60.0):
    """Temps pour passer de phi0 à ~5° sous l'action du seul GZ (corps rigide, sans houle).
    None si le bateau reste bloqué (position retournée stable)."""
    from scipy.integrate import solve_ivp
    gz_i = lambda p: np.interp(abs(p), phi, gz) * np.sign(p)
    I = added_mass_factor * I_roll
    w0 = math.sqrt(max(M * G * abs(gz_i(5.0) / math.radians(5.0)), 1e-6) / I)
    c = 2 * damping_ratio * I * w0

    def f(t, y):
        p, w = y
        return [w, (-M * G * gz_i(math.degrees(p)) - c * w) / I]

    ev = lambda t, y: y[0] - math.radians(5.0)
    ev.terminal = True
    sol = solve_ivp(f, [0, t_max], [math.radians(phi0_deg), 0.0], events=ev, max_step=0.02)
    return float(sol.t_events[0][0]) if len(sol.t_events[0]) else None


def check_stl(path: str) -> dict:
    """Contrôle rapide d'un STL externe (étanchéité, unités, volume)."""
    m = trimesh.load(path, force="mesh")
    rep = dict(
        file=path, n_faces=len(m.faces), n_vertices=len(m.vertices),
        watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent),
        n_bodies=len(m.split(only_watertight=False)), euler=int(m.euler_number),
        bounds_min=m.bounds[0].tolist(), bounds_max=m.bounds[1].tolist(),
        extents=m.extents.tolist(), volume_L=float(m.volume * 1000),
        volume_centroid=m.center_mass.tolist(),
    )
    L = m.extents[0]
    rep["unit_guess"] = "m" if 1.0 < L < 3.0 else ("mm" if 1000 < L < 3000 else "?")
    rep["ok"] = bool(rep["watertight"] and rep["n_bodies"] == 1 and rep["unit_guess"] == "m" and m.volume > 0)
    return rep
