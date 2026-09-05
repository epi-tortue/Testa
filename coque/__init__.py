"""Epi'Tortue - conception et optimisation d'une coque S1 (ailes inondables, sans quille).

Modules :
    params      constantes, variables libres (bornes + défauts), validité
    geometrie   Bézier rationnelle, loft d'anneaux -> maillage étanche
    mesh        CoqueMesh : coque paramétrique (section NURBS, plan de forme, rocker, pont bombé)
    ailes       Ailes : caissons latéraux paramétriques (largeur, épaisseur, bord épaissi, trous)
    masses      modèle de masses (structure calculée, charge utile, lest, panneaux)
    hydro       Carene : tirant d'eau au repos, surface mouillée, Cb, Cp
    gz          Vessel : carènes inclinées, courbe GZ 0-180°, métriques
    stabilite   verdict d'auto-redressement (ailes sèches -> noyées à l'immersion des trous)
    propulsion  résistance, puissance, énergie par km
    energie     production solaire de la plateforme
    objectif    évaluation complète d'un candidat -> score scalaire pour l'optimiseur
    optim       CMA-ES multi-départs avec reprise sur checkpoint
"""
