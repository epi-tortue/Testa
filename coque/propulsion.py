"""Puissance et énergie nécessaires pour faire avancer la coque.

    carène (hydro.Carene) -> surface mouillée, Lwl, Bwl, Cb
    vitesse V              -> résistance Rt(V) = hydrodynamique + aérienne (fardage)
    Rt, V, rendements      -> puissance à l'arbre, énergie sur une distance

Hypothèses :
  - eau calme, coque nue (pas d'appendices) ;
  - régime déplacement : ITTC-57 + facteur de forme (Watanabe) + terme de vague forfaitaire ;
  - régime planing : Savitsky (1964) ; entre les deux : interpolation linéaire ;
  - fardage : 0,5 rho_air Cd A_frontale (V + vent apparent)², A_frontale = œuvres mortes
    de la coque (largeur × hauteur émergée) + tranche des ailes.
Le seul terme réellement empirique est la résistance de vague (c_vague / r_hump_max) :
c'est le point à caler si une mesure existe.
"""
import numpy as np

from .hydro import Carene
from . import params as P

G = P.G


class Propulsion:
    """Résistance, puissance et énergie.

    mesh          : trimesh de la coque (étanche)
    masse         : [kg] déplacement total
    x_cg          : [m] abscisse du centre de gravité (x = 0 à l'étrave)
    deadrise_deg  : [°] V du fond au CG (paramètre DEADRISE)
    eta_prop      : rendement hélice + coque ; eta_chaine : moteur + variateur + réducteur
    k_forme       : facteur de forme imposé ; None = Watanabe depuis la carène
    c_vague, r_hump_max : terme de vague forfaitaire en régime déplacement
    b_chine       : [m] largeur au bouchain pour Savitsky ; None = Bwl
    dCf           : supplément de rugosité
    vent          : [m/s] vent apparent moyen pour le fardage (0 = ignoré)
    aire_frontale_extra : [m2] surface frontale ajoutée aux œuvres mortes (ailes)
    """

    def __init__(self, mesh, masse, x_cg, deadrise_deg,
                 rho=P.RHO_EAU, nu=P.NU_EAU,
                 eta_prop=P.RENDEMENT_PROP, eta_chaine=P.RENDEMENT_CHAINE,
                 k_forme=None, c_vague=0.008, r_hump_max=0.20,
                 b_chine=None, dCf=0.0,
                 vent=0.0, cd_air=P.CD_AIR, aire_frontale_extra=0.0):
        self.carene = Carene(mesh, masse, rho)
        self.masse, self.x_cg, self.beta = masse, x_cg, float(deadrise_deg)
        self.rho, self.nu = rho, nu
        self.eta_prop, self.eta_chaine = eta_prop, eta_chaine
        self.k_forme, self.c_vague, self.r_hump_max = k_forme, c_vague, r_hump_max
        self.b_chine, self.dCf = b_chine, dCf
        self.vent, self.cd_air, self.aire_frontale_extra = vent, cd_air, aire_frontale_extra

    # -- frottement ----------------------------------------------------------------
    @staticmethod
    def _cf(Re):
        """ITTC-57."""
        return 0.075 / (np.log10(max(Re, 1.0e5)) - 2.0) ** 2

    def _b(self):
        return self.b_chine if self.b_chine else self.carene.resoudre()["Bwl"]

    # -- fardage ---------------------------------------------------------------------
    def aire_frontale(self):
        """Œuvres mortes vues de face : largeur maxi × hauteur émergée (× 0,85 de forme) + extra."""
        h = self.carene.resoudre()
        m = self.carene.mesh
        largeur = float(m.bounds[1, 1] - m.bounds[0, 1])
        h_emergee = max(0.0, float(m.bounds[1, 2]) - h["z_flottaison"])
        return 0.85 * largeur * h_emergee + self.aire_frontale_extra

    def _r_air(self, V):
        if self.vent <= 0 and V <= 0:
            return 0.0
        return 0.5 * P.RHO_AIR * self.cd_air * self.aire_frontale() * (V + self.vent) ** 2

    # -- régime déplacement -------------------------------------------------------------
    def _r_deplacement(self, V):
        h = self.carene.resoudre()
        Re = V * h["Lwl"] / self.nu
        Rf = 0.5 * self.rho * h["S"] * V * V * (self._cf(Re) + self.dCf)
        if self.k_forme is not None:
            k = self.k_forme
        else:   # Watanabe
            k = -0.095 + 25.6 * h["Cb"] / ((h["Lwl"] / h["Bwl"]) ** 2 * np.sqrt(h["Bwl"] / h["T"]))
            k = float(np.clip(k, 0.0, 0.5))
        # terme de vague FORFAITAIRE : croît en Fn^4 puis sature à la "bosse" (r_hump_max)
        Fnv = V / np.sqrt(G * h["volume"] ** (1.0 / 3.0))
        x = self.c_vague * Fnv ** 4 / self.r_hump_max
        Rw = self.masse * G * self.r_hump_max * x / (1.0 + x)
        return (1.0 + k) * Rf + Rw

    # -- régime planing (Savitsky 1964) ----------------------------------------------------
    def _r_savitsky(self, V):
        h = self.carene.resoudre()
        b = self._b()
        W = self.masse * G
        beta = self.beta
        lcg = h["x_tableau"] - self.x_cg
        if lcg <= 0:
            raise ValueError("x_cg doit être en avant du tableau arrière")
        Cv = V / np.sqrt(G * b)
        CLb = W / (0.5 * self.rho * V * V * b * b)
        CL0 = CLb
        for _ in range(100):
            CL0 = CLb + 0.0065 * beta * CL0 ** 0.60
        LAM_MAX = 12.0

        def lam(tau_deg):
            f = lambda L: (tau_deg ** 1.1 * (0.0120 * np.sqrt(L) + 0.0055 * L ** 2.5 / Cv ** 2) - CL0)
            lo, hi = 1e-6, LAM_MAX
            if f(hi) < 0:
                return None, +1.0
            if f(lo) > 0:
                return None, -1.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) < 0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi), None

        def residu(tau_deg):
            L, borne = lam(tau_deg)
            if L is None:
                return borne * 1e3
            lp = (0.75 - 1.0 / (5.21 * Cv ** 2 / L ** 2 + 2.39)) * L * b
            return lp - lcg

        lo, hi = 0.2, 25.0
        f_lo, f_hi = residu(lo), residu(hi)
        if f_lo * f_hi > 0:
            return None
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = residu(mid)
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        tau = 0.5 * (lo + hi)
        L, _ = lam(tau)
        if L is None:
            return None
        tr, br = np.radians(tau), np.radians(beta)
        Vm = V * np.sqrt(max(1e-6, 1.0 - (0.0120 * np.sqrt(L) * tau ** 1.1) / (L * np.cos(tr))))
        Cf = self._cf(Vm * L * b / self.nu) + self.dCf
        Df = self.rho * Vm ** 2 * L * b * b * Cf / (2.0 * np.cos(br))
        Rt = W * np.tan(tr) + Df / np.cos(tr)
        valide = (L <= 4.0) and (2.0 <= tau <= 15.0)
        return dict(Rt=Rt, tau=tau, lam=L, Cv=Cv, valide=valide, S_mouillee=L * b * b / np.cos(br))

    # -- sélection de régime ----------------------------------------------------------------
    def resistance(self, V):
        """Résistance totale [N] à la vitesse V [m/s] (hydro + air), régime, détail Savitsky."""
        Cv = V / np.sqrt(G * self._b())
        R_dep = self._r_deplacement(V)
        R_air = self._r_air(V)
        if Cv <= 1.0:
            return R_dep + R_air, "déplacement", None
        sav = self._r_savitsky(V)
        if sav is None or sav["lam"] > 8.0:
            return R_dep + R_air, "déplacement (Savitsky sans solution)", None
        if Cv >= 2.0:
            note = "planing" if sav["valide"] else "planing (hors domaine strict)"
            return sav["Rt"] + R_air, note, sav
        w = Cv - 1.0
        return (1.0 - w) * R_dep + w * sav["Rt"] + R_air, "transition (interpolation)", sav

    # -- résultats -----------------------------------------------------------------------------
    def puissance(self, V):
        Rt, regime, sav = self.resistance(V)
        Pe = Rt * V
        Pa = Pe / self.eta_prop
        Pin = Pa / self.eta_chaine
        out = dict(V=V, Rt=Rt, R_air=self._r_air(V), P_effective=Pe, P_arbre=Pa, P_absorbee=Pin,
                   regime=regime)
        if sav:
            out.update(assiette_deg=sav["tau"], lambda_=sav["lam"])
        return out

    def energie(self, V, distance):
        """Énergie [J et Wh] pour parcourir `distance` [m] à la vitesse V."""
        r = self.puissance(V)
        E = r["P_absorbee"] * distance / V
        r.update(distance=distance, duree_s=distance / V, E_J=E, E_Wh=E / 3600.0)
        return r
