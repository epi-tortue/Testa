"""
================================================================================
 Epi'Tortue - Paramètres de conception (coque S1 : ailes inondables, sans quille)
================================================================================
Bateau autonome électrique < 2,4 m (Microtransat), coque à DÉPLACEMENT,
AUTO-REDRESSANTE PAR LA FORME (solution S1, façon aXatlantic) :
    coque étroite + ailes latérales creuses percées de trous + lest interne.
Référence : https://epi-tortue.github.io/docs/solutions/s1-coque/

TROIS CATÉGORIES
----------------
1) INTOUCHABLES   -> valeur fixe. Le programme ne les change JAMAIS.
2) TOUCHABLES     -> tuple (min, max). L'optimiseur explore cet intervalle.
                     Trois groupes : forme de COQUE (arguments de CoqueMesh),
                     AILES (arguments de Ailes), MASSES (lest).
3) CALCULÉS       -> mesurés sur le maillage (tirant d'eau, franc-bord, GZ...).
                     On ne les choisit pas, on les CONTRÔLE.

Unités SI : mètres, kilogrammes, secondes, degrés si précisé.
"""
from math import radians, tan, cos

# ##############################################################################
#  1) PARAMÈTRES INTOUCHABLES
# ##############################################################################

# ---- 1.1 Physique ------------------------------------------------------------
RHO_EAU  = 1025.0      # eau de mer                                    [kg/m3]
RHO_AIR  = 1.225       # air                                           [kg/m3]
G        = 9.81        # pesanteur                                     [m/s2]
NU_EAU   = 1.19e-6     # viscosité cinématique eau de mer ~15 °C       [m2/s]

# ---- 1.2 Règlement Microtransat + choix S1 -----------------------------------
LOA_MAX          = 2.40   # longueur hors-tout maximale (règlement)     [m]
B_HORS_TOUT_MAX  = 0.80   # largeur hors-tout ailes comprises (S1)       [m]
                          # (le règlement ne borne pas la largeur ; 0,80 m
                          #  est la cible S1 : plateforme solaire ~1,15 m²)

# ---- 1.3 Matériaux (sandwich mousse / verre-époxy) ---------------------------
RHO_MOUSSE     = 45.0     # âme mousse rigide PIR/PU                    [kg/m3]
RHO_STRATIFIE  = 1600.0   # stratifié verre/époxy                       [kg/m3]
EP_STRATIFIE   = 0.003    # épaisseur stratifié de coque                [m]
EP_MOUSSE      = 0.020    # épaisseur âme mousse de coque               [m]
M_SURF_COQUE   = EP_STRATIFIE * RHO_STRATIFIE + EP_MOUSSE * RHO_MOUSSE  # [kg/m2] ≈ 5,7
M_SURF_AILE    = 2.5      # caisson d'aile : peau verre légère + âme    [kg/m2]
M_SURF_SOLAIRE = 2.0      # panneaux souples + câblage, par m² de pont  [kg/m2]

# ---- 1.4 Budget de masse embarquée (matériel à transporter) -------------------
M_BATTERIES    = 8.0      # pack LiFePO4 (~480 Wh + marge)              [kg]
M_MOTEUR       = 1.5      # moteur brushless + pod                      [kg]
M_ELECTRONIQUE = 2.0      # calculateur, capteurs, Iridium, AIS         [kg]
M_DIVERS       = 1.5      # câblage, fixations, étanchéité, marge       [kg]
# Structure (coque + ailes) et panneaux : CALCULÉS sur la géométrie (masses.py).
# Lest interne : VARIABLE (cat. 2, groupe MASSES).

# ---- 1.5 Exigences de mission ---------------------------------------------------
# Tous les critères GZ sont évalués avec KG relevé de MARGE_KG (incertitude de devis de
# masse : ~300 g oubliés sur le pont) : GZ_robuste(phi) = GZ(phi) - MARGE_KG sin(phi).
MARGE_KG        = 0.010   # marge sur la hauteur du centre de gravité     [m]
FRANC_BORD_MINI = 0.08    # franc-bord mini au repos (réserve)          [m]
MARGE_GZ_MIN    = 0.030   # GZ >= 3 cm sur [90°,170°] ailes inondées    [m]
                          # (1 cm = 3,5 N.m pour 36 kg : rien ne garantit
                          #  le retour à l'endroit avec une aile mal vidée)
GM0_MIN         = 0.03    # stabilité initiale mini (état réel, ailes sèches) [m]
                          # (plancher ; en pratique c'est la gîte sous
                          #  VENT_GITE qui dimensionne la raideur)
PLAGE_GZ_MIN    = (90.0, 170.0)   # plage d'angles où la marge s'applique [°]
PHI_INONDATION_MIN = 25.0 # gîte mini avant immersion des trous de l'aile basse [°]
                          # (en dessous, l'aile sous le vent se remplit au
                          #  premier coup de gîte et la coque perd sa raideur)
# Tenue au vent (vent.py) : le plateau solaire est une voile dès que le bateau gîte.
CD_FARDAGE      = 1.1     # Cd de plaque plane (œuvres mortes + plateau)  [-]
VENT_MOYEN      = 7.0     # vent vrai moyen Atlantique N. -> gîte moyenne,
                          # production solaire × cos(gîte)               [m/s]
VENT_GITE       = 10.0    # vent de calcul (~20 nds, la moitié du temps) [m/s]
GITE_VENT_MAX   = 12.0    # gîte statique maxi sous VENT_GITE            [°]
MARGE_INOND_VENT = 5.0    # l'aile basse reste hors d'eau sous VENT_GITE
                          # avec cette marge d'angle                     [°]
AIRE_GZ_MIN     = 0.015   # aire sous GZ_robuste de 0 à PHI_AIRE (énergie
                          # de gîte / Mg : réserve dynamique en vagues)  [m.rad]
PHI_AIRE        = 60.0    # borne de l'aire dynamique                    [°]
# Piège ailes sèches : après un chavirage, les ailes mettent des minutes à se remplir.
# Tant qu'elles portent, la courbe "état A" peut avoir un équilibre STABLE entre 90° et
# 180° (bateau couché sur une aile). Il n'en sort que si, dans cette position, les trous
# d'une aile sont franchement sous l'eau : l'aile se remplit, on retombe sur la courbe
# réelle. Sinon c'est un piège définitif.
PROFONDEUR_TROU_MIN = 0.02  # immersion mini des trous à chaque équilibre stable
                            # ailes sèches de [90°,180°]                  [m]
PHI_PIEGE_MIN   = 90.0      # début de la plage examinée                 [°]

# ---- 1.6 Énergie (objectif) ------------------------------------------------------
SOLAIRE_WH_M2_JOUR = 615.0  # production moyenne par m² de panneau et par jour
                            # (plt/pont/pont_solaire.py, route Atlantique N.) [Wh/m2/j]
TAUX_COUVERTURE    = 0.85   # fraction du pont réellement couverte de cellules [-]
V_CROISIERE        = 1.0    # vitesse de référence du bilan énergie        [m/s]
RENDEMENT_PROP     = 0.50   # hélice + interaction coque                   [-]
RENDEMENT_CHAINE   = 0.85   # moteur + variateur + réducteur               [-]
VENT_APPARENT      = 2.0    # vent apparent moyen pour la traînée aérienne [m/s]
                            # (0 = ignorer le fardage ; valeur à caler)
CD_AIR             = 0.8    # coefficient de traînée aérienne des œuvres mortes [-]

# ---- 1.7 Ailes : ce qui n'est PAS optimisé ------------------------------------------
AILE_X0_FRAC   = 0.13     # début des ailes, fraction de L depuis l'étrave [-]
AILE_X1_FRAC   = 0.98     # fin des ailes                                  [-]
TROUS_Z_FRAC   = 1.0      # hauteur des trous sur la face extérieure de l'aile :
                          # 1 = bord supérieur (CDC §4.6), 0 = bord inférieur [-]
AILE_JEU       = 0.002    # jeu coque/aile pour éviter tout chevauchement  [m]

# ---- 1.8 Discrétisation (précision, pas la forme) --------------------------------
N_SECTIONS = 61
M_FOND     = 20
M_BOUCHAIN = 80
M_BORDE    = 20
M_PONT     = 9
N_SECTIONS_AILE = 25
PAS_GZ_OPTIM    = 10.0    # pas de la courbe GZ pendant l'optimisation    [°]
PAS_GZ_RAPPORT  = 5.0     # pas pour les rapports / tracés                [°]
PHI_GM0         = 2.0     # gîte de mesure de GM0, hors grille GZ         [°]
                          # (GM0 = GZ(PHI_GM0)/PHI_GM0 ; un GM0 tiré du 1er
                          #  point de la grille à 10° confondait la pente à
                          #  l'origine avec l'appui de l'aile sur l'eau)
MARGE_GRILLE    = 0.002   # surcote des seuils GZ/GM0 quand pas > PAS_GZ_RAPPORT [m]
                          # (l'optimiseur se cale pile sur les seuils ; la
                          #  grille grossière rate les creux entre deux points)
N_BISSECT_INOND = 3       # affinage par bissection de l'angle d'inondation
                          # (précision = pas / 2**N)
MARGE_GRILLE_DEG = PAS_GZ_RAPPORT / 2 ** N_BISSECT_INOND
                          # surcote du seuil d'inondation sur la grille grossière [°]
                          # = résolution de la bissection au pas fin : un candidat
                          #   accepté sur la grille l'est aussi au pas fin

# ##############################################################################
#  2) PARAMÈTRES TOUCHABLES  {nom: (min, max)}
# ##############################################################################

# ---- 2.1 Forme de coque : arguments de CoqueMesh ------------------------------------
VARIABLES_COQUE = {
    "L_COQUE":       (1.00, 2.35),   # longueur hors-tout                [m]
    "B_MAX":         (0.25, 0.50),   # largeur de coque au pont          [m]
                                     # (S1 : ~0,45 ; CDC : 0,25 ; les ailes
                                     #  complètent jusqu'à B_HORS_TOUT_MAX)
    "CREUX":         (0.15, 0.55),   # hauteur quille -> pont            [m]
    "DEADRISE":      (0.0, 45.0),    # V du fond                         [°]
    "FLARE":         (-10.0, 30.0),  # évasement (négatif = tumblehome)  [°]
    "F_BOUCHAIN":    (0.05, 0.95),   # étendue de l'arrondi du bouchain  [-]
    "W_BOUCHAIN":    (0.20, 3.00),   # poids NURBS du bouchain           [-]
    "X_MAITRE":      (0.40, 0.70),   # position du maître-bau            [-]
    "REMPL_AV":      (0.30, 0.80),   # remplissage avant                 [-]
    "REMPL_AR":      (0.30, 0.90),   # remplissage arrière               [-]
    "B_ETRAVE":      (0.02, 0.30),   # largeur résiduelle à l'étrave     [frac B_MAX]
    "B_TABLEAU":     (0.30, 1.00),   # largeur au tableau                [frac B_MAX]
    "ROCKER_AV":     (0.000, 0.150), # relèvement de quille à l'étrave   [m]
    "ROCKER_AR":     (0.000, 0.100), # relèvement de quille au tableau   [m]
    "HAUTEUR_BOMBE": (0.030, 0.200), # flèche du pont bombé = "pont intérieur"
                                     # (mini 3 cm : un pont plat n'est pas
                                     #  un pont intérieur, S1 §3)
                                     # S1 : volume étanche qui porte seul
                                     # quand les ailes sont noyées       [m]
}

# ---- 2.2 Ailes : arguments de Ailes (S1 §7 : volume + épaississement du bord) ----
VARIABLES_AILES = {
    "AILE_LARGEUR":   (0.05, 0.30),  # largeur d'une aile                [m]
    "AILE_EPAISSEUR": (0.04, 0.10),  # épaisseur à l'emplanture          [m]
                                     # (mini 4 cm : caisson sandwich de
                                     #  0,27 × 1,8 m ; 2 cm n'est pas
                                     #  constructible)
    "AILE_BORD":      (0.01, 0.08),  # sur-épaisseur du bord extérieur   [m]
                                     # (mini 1 cm : nervure de rive)
}

# ---- 2.3 Masses ------------------------------------------------------------------
VARIABLES_MASSE = {
    "LEST":           (0.0, 10.0),   # lest interne au fond (S1 : 5-10)  [kg]
}

VARIABLES_LIBRES = {**VARIABLES_COQUE, **VARIABLES_AILES, **VARIABLES_MASSE}

# Coque de référence : point de départ de l'optimiseur.
DEFAUTS = {
    "L_COQUE": 2.35, "B_MAX": 0.45, "CREUX": 0.30,
    "DEADRISE": 22.0, "FLARE": 10.0, "F_BOUCHAIN": 0.45, "W_BOUCHAIN": 0.60,
    "X_MAITRE": 0.58, "REMPL_AV": 0.55, "REMPL_AR": 0.55,
    "B_ETRAVE": 0.04, "B_TABLEAU": 0.70,
    "ROCKER_AV": 0.075, "ROCKER_AR": 0.025,
    "HAUTEUR_BOMBE": 0.03,
    "AILE_LARGEUR": 0.175, "AILE_EPAISSEUR": 0.04, "AILE_BORD": 0.01,
    "LEST": 4.0,
}

MAILLAGE_FIXE = {
    "N_SECTIONS": N_SECTIONS, "M_FOND": M_FOND, "M_BOUCHAIN": M_BOUCHAIN,
    "M_BORDE": M_BORDE, "M_PONT": M_PONT,
}

# ##############################################################################
#  3) CONTRÔLES CALCULÉS  (plages "saines", coque à déplacement lente)
# ##############################################################################
PLAGE_CP = (0.52, 0.68)
PLAGE_CB = (0.30, 0.55)


# ##############################################################################
#  4) OUTILS : curseur <-> valeurs, découpage, validité
# ##############################################################################

def curseur_vers_valeurs(curseur):
    """Curseur normalisé [0,1]^n (ordre de VARIABLES_LIBRES) -> dict de valeurs."""
    return {k: lo + (hi - lo) * float(c)
            for (k, (lo, hi)), c in zip(VARIABLES_LIBRES.items(), curseur)}


def valeurs_vers_curseur(valeurs):
    """dict de valeurs -> curseur normalisé (0,5 si l'intervalle est nul)."""
    out = []
    for k, (lo, hi) in VARIABLES_LIBRES.items():
        out.append(0.5 if hi == lo else (valeurs[k] - lo) / (hi - lo))
    return out


def separer(valeurs):
    """Découpe un dict complet en (kwargs coque, kwargs ailes, kwargs masse)."""
    return ({k: valeurs[k] for k in VARIABLES_COQUE},
            {k: valeurs[k] for k in VARIABLES_AILES},
            {k: valeurs[k] for k in VARIABLES_MASSE})


def section_valide(B_MAX, CREUX, DEADRISE, FLARE, marge=0.05):
    """Le fond (angle DEADRISE) et le bordé (angle FLARE) doivent se croiser
    ENTRE la quille et le livet, sinon la section est dégénérée sans qu'aucune
    erreur ne soit levée. Condition : tan(DEADRISE) * B/2 < CREUX."""
    if tan(radians(DEADRISE)) * B_MAX / 2.0 >= CREUX * (1.0 - marge):
        return False
    if cos(radians(DEADRISE + FLARE)) <= 1e-6:      # fond et bordé parallèles
        return False
    return True


def candidat_valide(p):
    """Contraintes DURES (géométrie impossible ou règlement) -> (bool, message).
    Les contraintes SOUPLES (largeur hors-tout, franc-bord) sont pénalisées
    de façon continue dans objectif.py, pas rejetées ici."""
    for nom, (lo, hi) in VARIABLES_LIBRES.items():
        if nom not in p:
            return False, f"{nom} manquant"
        if not (lo <= p[nom] <= hi):
            return False, f"{nom} = {p[nom]} hors de [{lo}, {hi}]"
    if p["L_COQUE"] > LOA_MAX:
        return False, "L_COQUE dépasse LOA_MAX (Microtransat)"
    if not section_valide(p["B_MAX"], p["CREUX"], p["DEADRISE"], p["FLARE"]):
        return False, "section dégénérée (DEADRISE trop fort pour B_MAX/CREUX)"
    return True, "OK"


def largeur_hors_tout(p):
    return p["B_MAX"] + 2.0 * p["AILE_LARGEUR"]


if __name__ == "__main__":
    ok, msg = candidat_valide(DEFAUTS)
    print(f"Coque de référence : {msg}   largeur hors-tout = {largeur_hors_tout(DEFAUTS):.2f} m")
    print(f"Variables libres : {len(VARIABLES_LIBRES)}")
    for nom, (lo, hi) in VARIABLES_LIBRES.items():
        val = DEFAUTS[nom]
        pos = (val - lo) / (hi - lo)
        print(f"  {nom:<15} {lo:>7.3f} |{'-' * int(pos * 20):<20}| {hi:>7.3f}   défaut = {val}")
