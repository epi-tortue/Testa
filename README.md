# Testa - conception de coque

Outil de conception et d'optimisation de la coque d'un bateau autonome électrique pour le
[Microtransat](https://www.microtransat.org/) (LOA < 2,4 m), selon la solution
[S1 - coque à ailes inondables](https://epi-tortue.github.io/docs/solutions/s1-coque/) :
coque à déplacement **auto-redressante par la forme**, sans quille lestée, avec deux
**ailes latérales creuses** percées de trous qui portent les panneaux solaires, un
**pont bombé étanche** qui joue le rôle du pont intérieur, et un **lest interne**.

L'objectif est de trouver la géométrie (coque + ailes + lest) qui **maximise l'énergie
solaire produite par rapport à l'énergie de propulsion**, sous contrainte
d'auto-redressement depuis 180° ailes noyées, de franc-bord et de largeur hors-tout.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
python3 -m coque.params                 # bornes et défauts des 19 variables
python3 -m coque.mesh                   # maillage de la coque de référence + contrôle d'étanchéité
python3 resolve.py --test               # rapport complet de la coque de référence -> outputs/test_*
python3 resolve.py --test --valeurs x.json
python3 resolve.py                      # optimisation CMA-ES multi-départs (reprise automatique)
python3 chercher_coques_redressantes.py --n 256   # balayage Sobol -> CSV + top coques GO
python3 trace_top_go.py                 # courbes GZ réel / A / B des meilleures coques
pytest                                  # validation (formes analytiques, étanchéité, non-régression)
```

## Ce que calcule le score

```
score = production solaire [Wh/j] × cos(gîte moyenne sous 7 m/s)
        / énergie pour 1 km à 1 m/s [Wh/km]  -  pénalités
```

- **Production** : aire projetée du pont de coque + dessus des ailes, × taux de couverture,
  × 615 Wh/m²/j (moyenne route Atlantique N., voir `plt/pont/pont_solaire.py`), × cos de
  la gîte statique sous le vent moyen `VENT_MOYEN` : une coque molle navigue couchée et
  ses panneaux ne voient plus le ciel.
- **Énergie** : frottement ITTC-57 + facteur de forme + vague forfaitaire + fardage
  (vent apparent moyen `VENT_APPARENT`), rendements hélice et chaîne électrique.
- **Tenue au vent** (`vent.py`) : couple de gîte = pression dynamique × Cd × (œuvres
  mortes × cos φ + plateau solaire × sin φ) × bras, comparé à M·g·GZ(φ). La gîte
  d'équilibre sous `VENT_GITE` (10 m/s, ~20 nds) doit rester sous `GITE_VENT_MAX` (12°)
  et sous l'angle d'inondation moins `MARGE_INOND_VENT` (5°). C'est ce critère qui
  dimensionne la raideur initiale, plus que le plancher GM0.
- **Robustesse** : tous les critères GZ sont évalués avec KG relevé de `MARGE_KG` (1 cm,
  soit ~300 g oubliés sur le pont) : GZ_rob(φ) = GZ(φ) − 1 cm × sin φ.
- **Pénalités** (continues, 200 points par cm de déficit, 20 points par degré ou par
  mm·rad) : GZ_rob < 3 cm sur [90°, 170°], GZ_rob < 0 quelque part (grille + point juste
  après l'inondation de chaque aile), GM0 < 3 cm (équilibre dédié à 2°, indépendant du pas
  de la grille), aire sous GZ_rob de 0 à 60° < 15 mm·rad (réserve dynamique en vagues),
  gîte sous 10 m/s > 12°, aile basse noyée sous 10 m/s, aile basse inondée avant 25° de
  gîte (angle encadré par bissection, la contrainte porte sur la borne basse de
  l'intervalle), franc-bord < 8 cm, ailes dans l'eau au repos, largeur hors-tout > 0,80 m.
  Géométrie impossible / maillage non étanche / coque qui coule : score −10 000 (toujours
  pire qu'une coque valide).
- **Bornes constructibles** : ailes ≥ 4 cm d'épaisseur, nervure de rive ≥ 1 cm, pont
  bombé ≥ 3 cm. Un optimiseur qui ne paie pas la structure met tout à zéro.
- **Piège ailes sèches** : après un chavirage, les ailes mettent des minutes à se remplir
  (vidange calculée ≈ 150 s pour 12 trous de 8 mm). Tant qu'elles portent, la courbe
  « état A » a en général un équilibre stable vers 150–160° (bateau couché sur une aile).
  À chaque équilibre stable de l'état A sur [90°,180°], les trous d'une aile doivent être
  à ≥ `PROFONDEUR_TROU_MIN` (2 cm) sous l'eau, sinon rien ne remplit l'aile et le bateau
  y reste : 200 points par cm manquant. Le rapport liste ces positions. Corollaire de
  construction : des **évents** sur la face intérieure des ailes, sinon l'air ne sort pas.
- **Assiette** : l'équilibre au repos est toujours à assiette libre ; en « assiette
  bloquée » (grille de l'optimiseur) les autres gîtes sont calculées à l'assiette du
  repos, pas à 0. Tirant d'eau, franc-bord (au livet du maître-bau) et garde des ailes
  (point le plus bas des caissons) sont mesurés sur ce plan de flottaison réel.
- **Coque GO de référence** : `references/go_reference.json` est GO sous tous ces
  critères (GM0 7,4 cm, gîte 9° sous 10 m/s, inondation à 29°, GZ_min[90°,170°] 3,2 cm,
  score ≈ 309). Point de départ conseillé : `python3 resolve.py --valeurs
  references/go_reference.json`. Sur 192 tirages Sobol aléatoires, aucun n'est GO : la
  région faisable est étroite, partir d'une coque GO économise beaucoup de budget.
- **Anti-artefact de grille** : pendant l'optimisation (pas 10°) les seuils GZ/GM0 sont
  surcotés de `MARGE_GRILLE` (2 mm) et le seuil d'inondation de `MARGE_GRILLE_DEG`
  (0,625°, la résolution de la bissection au pas fin) ; l'optimum de chaque run CMA-ES est ensuite revalidé
  au pas fin (5°, assiette libre) et c'est ce score-là qui classe les runs et qui est
  affiché. Un candidat qui n'est « GO » que sur la grille grossière ne peut plus gagner.
  GZ est aussi évalué juste après l'inondation de chaque aile (là où la courbe chute entre
  deux points de grille) et entre dans le contrôle GZ ≥ 0.
- **Deux classements** : la pénalité étant continue, un quasi-GO peut avoir un meilleur
  score qu'un vrai GO. `resolve.py` exporte donc en priorité la meilleure coque GO au pas
  fin (préfixe `optim`) et, si le meilleur score fin est une autre coque NO-GO, l'exporte
  aussi (préfixe `optim_nogo`).

## Modèle de stabilité

La courbe GZ 0–180° est calculée par carènes inclinées sur les maillages (coque + deux
ailes). Les ailes portent tant que leurs **trous** (bord supérieur extérieur) sont hors
d'eau ; dès qu'un trou passe sous la flottaison à l'équilibre, l'aile est **noyée
définitivement** et ne porte plus (courbe "réelle", CDC §6). Les courbes pures A (ailes
sèches) et B (ailes noyées) sont disponibles pour les rapports. Un filtre rapide (GZ à
172° coque seule) élimine en 0,4 s les coques stables à l'envers.

## Organisation

```
coque/            package
  params.py       constantes, 19 variables libres (bornes + défauts), validité
  geometrie.py    Bézier rationnelle, loft d'anneaux -> maillage étanche orienté
  mesh.py         CoqueMesh : coque paramétrique
  ailes.py        Ailes : caissons (largeur, épaisseur, bord épaissi, trous)
  masses.py       modèle de masses (structure calculée, charge, lest, panneaux)
  hydro.py        Carene : tirant d'eau, surface mouillée, Cb, Cp
  gz.py           Vessel : équilibre incliné, courbe GZ, métriques
  stabilite.py    verdict d'auto-redressement (inondation séquentielle des ailes)
  propulsion.py   résistance, puissance, énergie
  energie.py      production solaire
  objectif.py     evaluer(valeurs) -> dict complet ; score(curseur) -> float
  optim.py        CMA-ES multi-départs + checkpoints
  export.py       STL, JSON, tracé GZ
resolve.py, chercher_coques_redressantes.py, trace_top_go.py   scripts
tests/            pytest
checkpoints/      état de l'optimisation en cours (reprise automatique)
```

## Variables libres (19)

| Groupe | Variables |
|---|---|
| Coque (15) | `L_COQUE`, `B_MAX`, `CREUX`, `DEADRISE`, `FLARE`, `F_BOUCHAIN`, `W_BOUCHAIN`, `X_MAITRE`, `REMPL_AV`, `REMPL_AR`, `B_ETRAVE`, `B_TABLEAU`, `ROCKER_AV`, `ROCKER_AR`, `HAUTEUR_BOMBE` |
| Ailes (3) | `AILE_LARGEUR`, `AILE_EPAISSEUR`, `AILE_BORD` (épaississement du bord extérieur) |
| Masses (1) | `LEST` (kg, au fond) |

Correspondance avec les 5 paramètres d'optimisation du CDC §7 : volume des ailes
(`AILE_LARGEUR`, `AILE_EPAISSEUR`), forme de section (`DEADRISE`, `FLARE`, bouchain),
pont intérieur (`HAUTEUR_BOMBE`), hauteur de G (`LEST`), bord des ailes (`AILE_BORD`).

## Hypothèses à caler

`M_SURF_AILE`, `M_SURF_SOLAIRE`, `VENT_APPARENT`, `SOLAIRE_WH_M2_JOUR`, le terme de vague
(`c_vague`, `r_hump_max`) et les positions des masses (`masses.py`) sont des estimations :
voir les commentaires de `coque/params.py`.
