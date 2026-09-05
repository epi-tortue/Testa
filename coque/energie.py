"""Bilan énergie : production solaire de la plateforme (pont de coque + ailes)."""
from .geometrie import aire_projetee_z
from . import params as P


def surface_panneaux(coque, ailes):
    """Surface de cellules [m2] : aire projetée du pont de coque + dessus des ailes,
    multipliée par le taux de couverture."""
    pont = aire_projetee_z(coque.mesh, coque.face_pont)
    return P.TAUX_COUVERTURE * (pont + ailes.aire_plan_totale())


def production_journaliere(coque, ailes):
    """Énergie moyenne produite par jour [Wh/j]."""
    return surface_panneaux(coque, ailes) * P.SOLAIRE_WH_M2_JOUR
