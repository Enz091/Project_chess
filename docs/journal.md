# Journal du projet

Détails d'implémentation, mesures et listes de tests, étape par étape.
`CLAUDE.md` (section Avancement) ne garde que les décisions et les infos nécessaires à une future
session ; tout le reste va ici. Ajouter une section à la fin de chaque étape.

## Étape 0 — Mise en place (2026-10-20)

- Python 3.12.10 et Git 2.55 installés via winget. Venv avec torch 2.14.1+cpu, chess 1.11.2, numpy, tqdm, pytest.
- `device.py` et `checkpoint.py` écrits. 3 tests verts, dont « reprise = run d'une traite ».
- `scripts/smoke_resume.py` : interruption à l'étape 60 puis reprise automatique vérifiée.

## Étape 1 — Encodage (2026-10-20)

- `encoding.py` : tenseur `(18, 8, 8)` float32 du point de vue du camp au trait (Noirs = `board.mirror()`,
  retournement vertical seulement, pas de miroir gauche-droite à cause des roques).
  Plans : 0-5 mes pièces, 6-11 pièces adverses, 12-15 roques (mon petit/grand, adverse petit/grand),
  16 en passant (seulement si la prise est légale), 17 compteur des 50 coups / 100 (plafonné à 1).
  Pas d'historique ni de répétitions pour l'instant.
- Coups : indice = `case_départ_retournée * 73 + plan` (4672). Plans 0-55 mouvements de dame
  (direction * 7 + distance - 1, directions N, NE, E, SE, S, SO, O, NO), 56-63 cavalier,
  64-72 sous-promotions (C/F/T × prise gauche/tout droit/prise droite). Promotion en dame = coup normal.
  API : `move_to_index(move, board)`, `index_to_move(index, board)` (le `board` donne le trait et sert à
  reconnaître la promotion en dame), `legal_move_mask(board)`.
- Stockage compact : 67 octets `uint8` par position (64 cases codées 0-12, masque de roques, case en passant,
  compteur 50 coups). `pack_board` / `unpack_batch` (vectorisé). Environ 270 Mo pour 4 M de positions.
  Les cibles (indice du coup en int16, résultat en int8) seront stockées à côté à l'étape 3.
- 11 tests verts : aller-retour de tous les coups légaux (>3000 positions aléatoires + FEN ciblées
  roque / en passant / promotions), unicité des indices, masque, symétrie `board` / `board.mirror()`
  (tenseur, masque et indices de coups), pack/unpack = encode.
- `scripts/show_encoding.py [--fen ...] [--all-planes]` : affiche les plans et les coups légaux avec leurs indices.

## Étape 2 — Réseau (2026-10-20)

- `model.py` : `ModelConfig` (n_blocks, n_filters, value_hidden ; `ModelConfig.named("tiny")` lit
  `configs/tiny.toml`), `ChessNet`, `build_model(nom_ou_config)`, `count_parameters`.
- Tronc : conv 3×3 + BN + ReLU, puis N `ResBlock` (conv-BN-ReLU-conv-BN, + x, ReLU).
  Tête politique : conv 3×3 + BN + ReLU puis conv 1×1 vers 73 plans ; sortie `(B,73,8,8)` permutée en
  `(B,8,8,73)` et aplatie, donc l'indice vaut `case_départ*73 + plan` comme `encoding.move_to_index` (testé).
  Tête valeur : conv 1×1 → 1 plan + BN + ReLU, FC 64→hidden→1, `tanh`.
  Le réseau renvoie des logits bruts `(B,4672)` et une valeur `(B,)` du point de vue du camp au trait.
- Paramètres : **tiny = 352 460** (4 × 64), **kaggle = 3 140 940** (10 × 128).
- Perte : `masked_log_softmax` / `masked_policy` (illégaux à -1e9, calculé en float32), `policy_value_loss` =
  entropie croisée (cible = indice du coup OU distribution de visites) + MSE valeur, poids 1 pour chacun.
  Retourne aussi les deux termes pour les logs.
- 8 tests dans `tests/test_model.py` (19 au total, tous verts) : configs, formes et plage de la valeur,
  correspondance de la sortie politique avec les indices de coups, probabilités légales = 1, perte indice =
  perte one-hot, logits illégaux ignorés, surapprentissage : tiny mémorise 50 positions à cibles aléatoires,
  perte 4,0 → ~1e-4 en ~50 pas Adam (lr 2e-3), 100 % de coups justes en mode eval (le test fait 100 pas).
- `requires-python` passé à `>=3.11` (les configs sont en TOML, lu avec `tomllib`).
- Vitesse d'inférence CPU, `scripts/bench_inference.py` (positions par seconde) :

  | lot | tiny, 14 threads | kaggle, 14 threads |
  |---|---|---|
  | 1 | 784 | 226 |
  | 8 | 1 804 | 436 |
  | 32 | 3 691 | 643 |
  | 64 | 4 778 | 666 |
  | 128 | 5 615 | 731 |
  | 256 | 5 875 | 622 |

  Avec 1 thread : tiny 568 (lot 1), 1 215 (lot 32), 1 279 (lot 128) ; kaggle 103, 153, 146.
  Avec 4 threads : tiny 838, 2 695, 3 773 ; kaggle 215, 435, 441.
  Conclusion : le lot rapporte × 4 à × 8, donc le MCTS devra évaluer par lots (étape 5) ;
  le réseau kaggle est trop lent pour du MCTS sur ce CPU, à réserver au GPU.

## Étape 3 — Données Lichess (2026-10-20)

- **Source retenue** : base mensuelle `standard/lichess_db_standard_rated_2026-09.pgn.zst` (29,2 Go compressé,
  taille vérifiée par requête HEAD), lue **en flux HTTP + décompression zstd à la volée**, jamais stockée ; on
  s'arrête dès la cible atteinte. Écartée : la base « Elite » (database.nikonoel.fr, ~80 Mo/mois en `.zip`) car
  elle ne contient que des joueurs 2500+ contre 2300+, trop étroit pour un seuil à 2000. Biais connu : on ne lit
  que les premiers jours du mois (ordre chronologique du fichier) ; pour plus de diversité, relancer sur un autre mois.
- **Pipeline** (`src/chesszero/data.py`) : le processus principal découpe les parties et filtre sur les **en-têtes
  seuls** (peu coûteux) ; un pool de workers (14 max) rejoue les parties gardées avec `chess.pgn`. Au plus
  `2 × workers` lots en vol (RAM bornée), résultats consommés dans l'ordre : la sortie ne dépend pas du nombre de
  workers (testé). Fenêtre zstd : `max_window_size=2**31` (les fichiers Lichess utilisent une grande fenêtre).
- **Filtres** (`configs/data.toml`, surchargeables en ligne de commande) : `min(WhiteElo, BlackElo) ≥ 2000` ;
  `Result` décisif ou nul ; `Termination` ∈ {Normal, Time forfeit} ; pas de `FEN`/`SetUp`/variante ; cadence
  estimée `base + 40 × incrément ≥ 180 s` (définition Lichess du bullet), correspondance (`-`) écartée ;
  parties de moins de 20 demi-coups écartées ; partie entière jetée si un coup est illisible/illégal.
- **Séparation** : `crc32(Site) % 10000 < 200` → validation (≈ 2 % des PARTIES). Déterministe, indépendant de
  l'ordre et du nombre de workers ; aucune partie dans les deux splits (vérifié sur les vraies données).
- **Format des shards** (`train_00000.npz`, `val_00000.npz`, 250 000 positions chacun, écriture atomique, `.npz` non
  compressé) : `boards (N,67) uint8`, `moves (N,) int16`, `results (N,) int8` (+1/0/−1 du point de vue du camp au
  trait), `games (N,) int32` (n° de partie dans la source). 71 octets par position. `manifest.json` : config,
  compteurs de rejets par raison, shards, octets lus/écrits, durée. Les positions restent dans l'ordre des
  parties : l'étape 4 doit mélanger (tout tient en RAM : 3 M × 71 o ≈ 215 Mo), `data.load_split(dossier, split)`.
- **Tests** (`tests/test_data.py`, 11 tests, 30 au total) sur un PGN synthétique au format Lichess (72 parties
  aléatoires avec commentaires `[%clk]`/`[%eval]`, dont des cas à rejeter : Elo bas, bullet, abandon, trop courte,
  FEN, résultat `*`, plus une partie à coup illégal) : compteurs de rejets exacts ; pour chaque position, à partir
  des **seules données stockées** → position valide (`Board.is_valid()`, un roi par camp, pas de pion en
  rangée 1/8) et coup légal ; relecture indépendante des parties → mêmes octets, mêmes indices, résultat du bon
  signe pour les deux camps ; train/val sans partie commune ; shards de taille fixe ; workers 1 vs 2 identiques ;
  lecture `.zst` + arrêt à la cible ; garde-fou d'écrasement. Test de mutation : inverser le signe du résultat
  fait bien échouer le test.
- **Essai** (1 000 parties, `data/essai`) : 61 parties gardées (6,1 %), 5 194 positions, 0,4 Mo, 2 s.
- **Extraction réelle** (`data/lichess_2000`, cible 3 M) :

  | | |
  |---|---|
  | Parties lues | 660 723 |
  | Rejetées (en-têtes) | Elo 553 759 · bullet 65 228 · time_control 97 · result 76 · termination 63 |
  | Rejetées après lecture | trop courtes 515 |
  | Parties gardées | 38 285 (5,8 % des lues) |
  | Positions | 3 002 978 (train 2 941 812 · val 61 166 = 2,04 %) |
  | Flux lu (compressé) | 213,8 Mo |
  | Taille sur disque | 222,2 Mo (13 shards) |
  | Durée | 39 s (≈ 77 000 pos/s, 14 workers) |

  Résultats équilibrés (−1 : 45 %, 0 : 9 %, +1 : 46 %), indice de coup max 4660 < 4672.
