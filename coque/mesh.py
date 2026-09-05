"""Coque paramétrique : section transversale (Bézier rationnelle), plan de forme,
rocker et pont bombé, lissés en un maillage trimesh étanche.

Repère : x = longitudinal (0 = étrave, L = tableau), y = transversal (tribord > 0),
z = vertical vers le haut (0 = point bas de la quille au maître-bau).

Usage :
    coque = CoqueMesh(L_COQUE=2.35, B_MAX=0.45, ...)
    mesh = coque.generate()          # trimesh.Trimesh, coque.face_pont = masque des faces du pont
    coque.controle_maillage()
"""
import numpy as np
import trimesh

from .geometrie import bezier_rationnelle, loft, maillage


class CoqueMesh:

    def __init__(
            self,
            # --- dimensions principales --------------------------------------
            L_COQUE=2.35,        # [m] longueur hors-tout
            B_MAX=0.45,          # [m] largeur maxi (au pont)
            CREUX=0.30,          # [m] hauteur quille -> pont
            # --- section transversale -----------------------------------------
            DEADRISE=22.0,       # [°] V du fond (0 = plat)
            FLARE=10.0,          # [°] évasement du bordé (0 = vertical, <0 = tumblehome)
            F_BOUCHAIN=0.45,     # [0-1] étendue de l'arrondi du bouchain
            W_BOUCHAIN=0.60,     # [>0] poids NURBS : <1 arrondi, >1 anguleux
            # --- évolution longitudinale ---------------------------------------
            X_MAITRE=0.58,       # [0-1] position du maître-bau (0 = étrave)
            REMPL_AV=0.55,       # [0-1] remplissage avant (petit = plein)
            REMPL_AR=0.55,       # [0-1] remplissage arrière (grand = plein)
            B_ETRAVE=0.04,       # [frac B_MAX] largeur résiduelle à l'étrave
            B_TABLEAU=0.70,      # [frac B_MAX] largeur au tableau
            ROCKER_AV=0.075,     # [m] relèvement de la quille à l'étrave
            ROCKER_AR=0.025,     # [m] relèvement de la quille au tableau
            # --- pont -----------------------------------------------------------
            HAUTEUR_BOMBE=0.01,  # [m] flèche du pont bombé au centre
            # --- discrétisation ---------------------------------------------------
            N_SECTIONS=61, M_FOND=20, M_BOUCHAIN=80, M_BORDE=20, M_PONT=9,
    ):
        self.L_COQUE, self.B_MAX, self.CREUX = L_COQUE, B_MAX, CREUX
        self.DEADRISE, self.FLARE = DEADRISE, FLARE
        self.F_BOUCHAIN, self.W_BOUCHAIN = F_BOUCHAIN, W_BOUCHAIN
        self.X_MAITRE, self.REMPL_AV, self.REMPL_AR = X_MAITRE, REMPL_AV, REMPL_AR
        self.B_ETRAVE, self.B_TABLEAU = B_ETRAVE, B_TABLEAU
        self.ROCKER_AV, self.ROCKER_AR = ROCKER_AV, ROCKER_AR
        self.HAUTEUR_BOMBE = HAUTEUR_BOMBE
        self.N_SECTIONS, self.M_FOND, self.M_BOUCHAIN = N_SECTIONS, M_FOND, M_BOUCHAIN
        self.M_BORDE, self.M_PONT = M_BORDE, M_PONT

        self.mesh = None
        self.face_pont = None
        self.X = None            # abscisses des stations
        self.B_loc = None        # demi-largeur au pont par station
        self.z_quille = None     # cote de la quille par station
        self.anneaux = None      # (N_SECTIONS, m, 3) contours de section

    # --- section de référence ------------------------------------------------

    @staticmethod
    def demi_section_gen(deadrise, flare, f_bouchain, w_bouchain, demi_largeur, creux,
                         m_fond=20, m_bouchain=80, m_borde=20):
        """Demi-section normalisée (y, z dans [0,1]) : quille K=(0,0) -> livet D=(1,1).
        Fond à l'angle deadrise, bordé à l'angle flare, coin C arrondi par une Bézier
        rationnelle de poids w_bouchain."""
        K = np.array([0.0, 0.0])
        D = np.array([demi_largeur, creux])
        d, f = np.radians(deadrise), np.radians(flare)
        u = np.array([np.cos(d), np.sin(d)])      # direction du fond
        v = np.array([np.sin(f), np.cos(f)])      # direction du bordé
        t_s = np.linalg.solve(np.column_stack([u, -v]), D - K)
        C = K + t_s[0] * u                        # coin théorique du bouchain
        B1 = C + f_bouchain * (K - C)
        B2 = C + f_bouchain * (D - C)
        fond = np.linspace(K, B1, m_fond)
        bouchain = bezier_rationnelle(B1, C, B2, w_bouchain, m_bouchain)
        borde = np.linspace(B2, D, m_borde)
        return np.vstack([fond, bouchain[1:], borde[1:]]) / D

    # --- génération -------------------------------------------------------------

    def generate(self):
        demi_B = self.B_MAX / 2.0
        x_m = self.X_MAITRE * self.L_COQUE

        profil = self.demi_section_gen(
            self.DEADRISE, self.FLARE, self.F_BOUCHAIN, self.W_BOUCHAIN, demi_B, self.CREUX,
            self.M_FOND, self.M_BOUCHAIN, self.M_BORDE)

        # plan de forme : demi-largeur le long de x
        avant = bezier_rationnelle(np.array([0.0, self.B_ETRAVE * demi_B]),
                                   np.array([self.REMPL_AV * x_m, demi_B]),
                                   np.array([x_m, demi_B]), 1.0, 200)
        arriere = bezier_rationnelle(np.array([x_m, demi_B]),
                                     np.array([x_m + self.REMPL_AR * (self.L_COQUE - x_m), demi_B]),
                                     np.array([self.L_COQUE, self.B_TABLEAU * demi_B]), 1.0, 200)
        plan = np.vstack([avant, arriere[1:]])

        X = np.linspace(0.0, self.L_COQUE, self.N_SECTIONS)
        B_loc = np.interp(X, plan[:, 0], plan[:, 1])
        z_quille = np.where(
            X < x_m,
            self.ROCKER_AV * ((x_m - X) / x_m) ** 2,
            self.ROCKER_AR * ((X - x_m) / (self.L_COQUE - x_m)) ** 2)
        h_loc = self.CREUX - z_quille

        # anneaux fermés : [tribord (quille -> livet) | pont | bâbord (livet -> quille)]
        n_droite, n_pont = len(profil), self.M_PONT - 2
        anneaux = []
        for i in range(self.N_SECTIONS):
            droite = np.column_stack([profil[:, 0] * B_loc[i], z_quille[i] + profil[:, 1] * h_loc[i]])
            gauche = droite[::-1][:-1] * np.array([-1, 1])
            y_pont = np.linspace(B_loc[i], -B_loc[i], self.M_PONT)[1:-1]
            fleche = self.HAUTEUR_BOMBE * (1.0 - (y_pont / max(B_loc[i], 1e-9)) ** 2)
            pont = np.column_stack([y_pont, self.CREUX + fleche])
            yz = np.vstack([droite, pont, gauche])
            anneaux.append(np.column_stack([np.full(len(yz), X[i]), yz]))
        anneaux = np.array(anneaux)

        vertices, faces, j_face = loft(anneaux, centre_sur_axe=True)
        j_pont = set(range(n_droite - 1, n_droite + n_pont))
        face_pont = np.array([j in j_pont for j in j_face], dtype=bool)

        mesh = maillage(vertices, faces)
        assert len(mesh.faces) == len(face_pont), "faces perdues au traitement : masque du pont invalide"
        mesh.face_pont = face_pont

        self.mesh, self.face_pont = mesh, face_pont
        self.X, self.B_loc, self.z_quille, self.anneaux = X, B_loc, z_quille, anneaux
        return mesh

    # --- exploitation ---------------------------------------------------------------

    def demi_largeur_max(self, z_min, z_max):
        """Par station : plus grande |y| du contour entre les cotes z_min et z_max
        (sert à accoler les ailes sans chevauchement, y compris avec du tumblehome)."""
        if self.anneaux is None:
            raise RuntimeError("generate() d'abord")
        out = np.zeros(self.N_SECTIONS)
        for i, ring in enumerate(self.anneaux):
            m = (ring[:, 2] >= z_min) & (ring[:, 2] <= z_max)
            out[i] = np.abs(ring[m, 1]).max() if m.any() else np.abs(ring[:, 1]).max()
        return out

    def get_area(self, mask=None):
        if self.mesh is None:
            raise RuntimeError("generate() d'abord")
        faces = self.mesh.faces if mask is None else self.mesh.faces[mask]
        return trimesh.triangles.area(self.mesh.vertices[faces]).sum()

    def controle_maillage(self):
        if self.mesh is None:
            raise RuntimeError("generate() d'abord")
        mesh = self.mesh
        n_ouvertes = len(trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1))
        print("\n=== Contrôle du maillage =============================")
        print(f"Sommets                : {len(mesh.vertices)}")
        print(f"Triangles              : {len(mesh.faces)}")
        print(f"is_watertight (étanche): {mesh.is_watertight}")
        print(f"is_volume (volume sûr) : {mesh.is_volume}")
        print(f"Arêtes ouvertes        : {n_ouvertes}")
        print(f"Euler number (2=sphère): {mesh.euler_number}")
        print(f"Volume enveloppe       : {mesh.volume*1000:.1f} L   aire {mesh.area:.3f} m²")
        if not mesh.is_watertight:
            print("!! Maillage NON étanche : les mesures n'ont aucun sens.")


if __name__ == "__main__":
    import time
    t = time.time()
    coque = CoqueMesh()
    coque.generate()
    print(f"généré en {time.time() - t:.2f} s")
    coque.controle_maillage()
