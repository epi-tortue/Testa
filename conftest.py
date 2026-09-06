# Présence suffisante : pytest ajoute ce dossier (la racine du dépôt) à sys.path,
# ce qui rend le paquet `coque` importable depuis tests/ quelle que soit la
# façon de lancer pytest (`pytest`, `python -m pytest`, depuis un autre CWD).
