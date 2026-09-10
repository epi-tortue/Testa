"""Ailes latérales S1 : deux caissons creux accolés au livet, percés de trous.

Section d'une aile (plan y-z, tribord) :

        livet ┌──────────────────┐  z = CREUX            (dessus = pont solaire)
              │                  │
              └────────┐         │  z = CREUX - AILE_EPAISSEUR    (emplanture)
                       └─────────┘  z = CREUX - AILE_EPAISSEUR - AILE_BORD   (bord épaissi)
              y_in               y_in + AILE_LARGEUR

Le bord intérieur suit le flanc de la coque station par station (largeur constante) ;
l'épaississement du bord extérieur (AILE_BORD) est la variable de "stabilité secondaire à
l'endroit" du CDC §7. Les trous d'inondation sont sur la face extérieure, à la hauteur
TROUS_Z_FRAC : quand ils passent sous l'eau, l'aile se remplit et ne porte plus (état B).

Construction (non maillé, mais indispensable au fonctionnement) :
  - les trous d'inondation sont sur la face EXTÉRIEURE, bord supérieur ;
  - il faut des ÉVENTS sur la face INTÉRIEURE (côté coque), bord supérieur, sinon l'air
    ne sort pas et l'aile ne se remplit jamais : le bateau reste couché sur une aile
    pleine d'air (stabilite.pieges_ailes_seches). Les mêmes évents laissent rentrer
    l'air à la vidange ;
  - trous et évents doivent rester dégagés (pas de grille fine : fouling, sel).
"""
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .geometrie import loft, maillage, aire_projetee_z
from . import params as P


@dataclass
class Ailes:
    largeur: float
    epaisseur: float
    bord: float = 0.0
    x0: float = 0.0
    x1: float = 0.0
    z_top: float = 0.0
    meshes: dict = field(default_factory=dict)      # {+1: tribord, -1: bâbord}
    trous: dict = field(default_factory=dict)       # {+1: point (x,y,z), -1: ...}
    _aire_plan: float = 0.0                         # dessus d'UNE aile [m2]

    # ------------------------------------------------------------------ construction
    @classmethod
    def depuis_coque(cls, coque, AILE_LARGEUR, AILE_EPAISSEUR, AILE_BORD=0.0,
                     x0_frac=P.AILE_X0_FRAC, x1_frac=P.AILE_X1_FRAC,
                     n_sections=P.N_SECTIONS_AILE, jeu=P.AILE_JEU):
        """Construit les deux caissons sur une CoqueMesh déjà générée."""
        L, z_top = coque.L_COQUE, coque.CREUX
        z_bot_in = z_top - AILE_EPAISSEUR
        z_bot_out = z_bot_in - AILE_BORD
        # flanc de coque le plus large sur la hauteur de l'aile -> pas de chevauchement
        y_flanc = coque.demi_largeur_max(z_bot_out - 0.005, z_top + 0.005) + jeu
        X = np.linspace(x0_frac * L, x1_frac * L, n_sections)
        y_in = np.interp(X, coque.X, y_flanc)
        y_out = y_in + AILE_LARGEUR

        a = cls(AILE_LARGEUR, AILE_EPAISSEUR, AILE_BORD, X[0], X[-1], z_top)
        for s in (+1, -1):
            rings = np.stack([
                np.column_stack([X, s * y_in, np.full_like(X, z_top)]),
                np.column_stack([X, s * y_out, np.full_like(X, z_top)]),
                np.column_stack([X, s * y_out, np.full_like(X, z_bot_out)]),
                np.column_stack([X, s * y_in, np.full_like(X, z_bot_in)]),
            ], axis=1)                                   # (n, 4, 3)
            v, f, _ = loft(rings)
            a.meshes[s] = maillage(v, f)
            i_mid = len(X) // 2
            z_trou = z_bot_out + P.TROUS_Z_FRAC * (z_top - z_bot_out)
            a.trous[s] = np.array([X[i_mid], s * y_out[i_mid], z_trou])
        a._aire_plan = float(np.trapezoid(y_out - y_in, X))
        return a

    # ------------------------------------------------------------------ mesures
    def volume(self, side=+1):
        return float(abs(self.meshes[side].volume))

    def volume_total(self):
        return self.volume(+1) + self.volume(-1)

    def aire_coque(self, side=+1):
        """Surface développée du caisson (masse de structure)."""
        return float(self.meshes[side].area)

    def aire_plan(self, side=+1):
        """Aire du dessus d'une aile (panneaux solaires)."""
        return self._aire_plan

    def aire_plan_totale(self):
        return 2.0 * self._aire_plan

    def aire_frontale(self):
        """Surface frontale des deux ailes (fardage)."""
        return 2.0 * self.largeur * (self.epaisseur + self.bord)

    def centroide(self, side=+1):
        return np.asarray(self.meshes[side].center_mass)

    def est_etanche(self):
        return all(m.is_watertight for m in self.meshes.values())

    def temps_vidange_s(self, n_trous=12, diam_mm=8.0, cd=0.6, obstruction=0.0):
        """Vidange par gravité d'une aile pleine, trous en point bas (Torricelli intégré) :
        t = S_aile / (Cd * A_trous) * sqrt(2 h0 / g). obstruction = fraction bouchée."""
        a = n_trous * np.pi * (diam_mm / 2000.0) ** 2 * (1.0 - obstruction)
        if a <= 0:
            return float("inf")
        h0 = self.epaisseur + self.bord
        return self._aire_plan / (cd * a) * np.sqrt(2.0 * h0 / P.G)
