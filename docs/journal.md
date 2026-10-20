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
  trait), `games (N,) int32` (n° de partie dans la source). `value_valid (N,) uint8` (ajouté ensuite, voir la révision plus bas). 75 octets par position. `manifest.json` : config,
  compteurs de rejets par raison, shards, octets lus/écrits, durée. Les positions restent dans l'ordre des
  parties : l'étape 4 doit mélanger (tout tient en RAM : 3 M × 75 o ≈ 225 Mo), `data.load_split(dossier, split)`.
- **Tests** (`tests/test_data.py`, 11 tests, 30 au total) sur un PGN synthétique au format Lichess (72 parties
  aléatoires avec commentaires `[%clk]`/`[%eval]`, dont des cas à rejeter : Elo bas, bullet, abandon, trop courte,
  FEN, résultat `*`, plus une partie à coup illégal) : compteurs de rejets exacts ; pour chaque position, à partir
  des **seules données stockées** → position valide (`Board.is_valid()`, un roi par camp, pas de pion en
  rangée 1/8) et coup légal ; relecture indépendante des parties → mêmes octets, mêmes indices, résultat du bon
  signe pour les deux camps ; train/val sans partie commune ; shards de taille fixe ; workers 1 vs 2 identiques ;
  lecture `.zst` + arrêt à la cible ; garde-fou d'écrasement. Test de mutation : inverser le signe du résultat
  fait bien échouer le test.
- **Essai** (1 000 parties, `data/essai`) : 61 parties gardées (6,1 %), 5 194 positions, 0,4 Mo, 2 s.
- **Extraction initiale** (cible 3 M, BOT inclus) : 660 723 parties lues, 38 285 gardées, 3 002 978 positions,
  222 Mo, 39 s. Remplacée par l'extraction ci-dessous.

### Révision de l'étape 3 : filtre BOT, Time forfeit, `value_valid` (2026-10-20)

- **Filtre BOT** : `FilterConfig.exclude_bots = True` par défaut (`configs/data.toml`, `--include-bots` pour le
  désactiver). Une partie est rejetée (raison `bot`) si `WhiteTitle` ou `BlackTitle` vaut `BOT`. Le test est
  fait en dernier : « bot » ne compte que des parties qui auraient été gardées sans lui. 924 parties BOT
  rejetées (2,4 % des parties qui passaient les autres filtres).
- **Part des « Time forfeit »** (nouveau compteur `kept_by_termination` dans `manifest.json`, affiché par
  `prepare_data.py`). Mesurée sur l'extraction initiale (en la relançant avec `--include-bots`, qui redonne
  exactement 38 285 parties) : **9 069 parties sur 38 285 (23,7 %)**, soit 837 782 positions sur 3 002 978
  (27,9 %). Sans BOT : 9 387 parties sur 38 962 (24,1 %), 865 667 positions sur 3 004 035 (28,8 %).
- **Choix : garder ces parties pour la politique, pas pour la valeur.** Raison : quand une partie se termine
  au drapeau, le résultat dit qui a manqué de temps, pas qui avait la meilleure position (souvent la
  position était équilibrée, ou gagnée par le perdant). C'est une cible de valeur biaisée : parmi les positions de
  parties « Time forfeit » il y a 3,8 % de nulles contre 10,2 % pour « Normal ». Le coup joué, lui, reste un
  bon exemple de politique. Ignorer ces 24 % de parties pour tout aurait coûté un quart des exemples de politique.
- **Champ `value_valid`** (`(N,) uint8`, dans chaque shard) : 1 si `Termination == "Normal"`, 0 si
  « Time forfeit ». `data.py` ne l'invente pas après coup : il est écrit à l'extraction, constant sur toute la partie.
  L'ancien format (sans ce champ) est refusé par `train()` avec un message qui demande de relancer l'extraction.
- **Perte** (`model.policy_value_loss(..., value_valid=None)`) : la perte valeur est `Σ valid·(v − cible)² / Σ valid`
  (moyenne sur les positions valides seulement ; 0 si le lot n'en contient aucune) ; la perte politique porte
  toujours sur toutes les positions. Une position à 0 n'envoie aucun gradient à la tête valeur, ni
  directement ni via le tronc. Même règle dans le suivi (`Meter`) : perte valeur d'entraînement et de
  validation = moyenne sur les positions valides, pondérée par leur nombre (et non par la taille du lot) ; la
  perte totale affichée = politique (toutes positions) + valeur (valides).
- **Tests ajoutés** : filtre BOT (`reject_reason`, et une partie BOT dans le PGN synthétique) ; compteur
  `kept_by_termination` = recompte indépendant des en-têtes ; `value_valid` correct pour chaque partie
  (test de mutation : inverser la condition fait échouer le test) ; **gradient de la tête valeur** (`test_model.py`) :
  changer la cible des positions invalides ne modifie ni la perte valeur ni aucun gradient de la tête ; le gradient
  est identique à celui d'un lot qui ne contiendrait que les positions valides ; sans position valide le gradient
  est nul alors que la politique apprend toujours (mutation : remplacer la moyenne pondérée par une moyenne simple
  fait échouer le test) ; moyenne pondérée du `Meter` ; ancien format de données refusé.
- **Extraction réelle** (`data/lichess_2000`, cible 3 M, BOT exclus, `value_valid`) :

  | | |
  |---|---|
  | Parties lues | 686 052 |
  | Rejetées (en-têtes) | Elo 574 941 · bullet 67 739 · bot 924 · time_control 102 · result 79 · termination 67 |
  | Rejetées après lecture | trop courtes 538 |
  | Parties gardées | 38 962 (5,7 % des lues) : Normal 29 575 (75,9 %), Time forfeit 9 387 (24,1 %) |
  | Positions | 3 004 035 (train 2 943 133 · val 60 902 = 2,03 %) |
  | Positions à `value_valid = 1` | train 2 095 600 (71,2 %) · val 42 768 (70,2 %) |
  | Flux lu (compressé) | 222,0 Mo |
  | Taille sur disque | 225,3 Mo |
  | Durée | 45 s |

  Résultats des positions valides : −1 : 44,6 %, 0 : 10,2 %, +1 : 45,2 %. Indice de coup max < 4672.

## Étape 4a — Pré-entraînement supervisé, réseau tiny en local (2026-10-20)

### Ce qui a été écrit

- `src/chesszero/train.py` (boucle et suivi), `scripts/train.py` (point d'entrée), `scripts/plot_curves.py`
  (courbes PNG), `scripts/show_policy.py` (les meilleurs coups du réseau pour une position),
  `tests/test_train.py`. Extensions : `encoding.board_from_packed` / `legal_masks_from_packed`,
  `checkpoint.py` accepte un `scaler`, section `[train]` dans `tiny.toml` et `kaggle.toml`.
- **Données** : `load_split` en RAM ; à l'époque `e`, permutation de `default_rng([seed, e])`, dernier lot
  incomplet écarté. Le lot du pas `s` ne dépend que de `s` : la reprise n'a aucun état de lecture à restaurer.
- **Masques de coups légaux** : la perte et les précisions ne portent que sur les coups légaux, donc il faut
  un masque par position. Reconstruire la position depuis le format compact et appeler python-chess donne ~7 000
  positions/s, plus lent que le réseau sur GPU. Un pool de processus (`num_workers`) calcule les masques
  quelques lots à l'avance ; les lots sont identiques avec ou sans pool (testé). Le sous-ensemble de
  validation calcule ses masques une seule fois.
- **Optimiseur** (valeurs standard, détaillées dans `configs/*.toml`) : AdamW, `lr` de pointe 1e-3 (tiny, lots de
  256) et 2e-3 (kaggle, lots de 1024) ; décroissance de poids 0,01 sur les poids de convolutions/couches
  linéaires seulement (pas BatchNorm ni biais) ; coupure du gradient à la norme 1 ; montée linéaire sur 300 pas
  (1000 en kaggle) car les moments d'Adam et les statistiques BatchNorm sont instables au début, puis
  cosinus jusqu'à 1 % du maximum.
- **Précision mixte** : fp16 + `GradScaler` seulement si le device est CUDA (`amp = "auto"`), perte toujours en
  float32. Les GPU de Kaggle (T4/P100) n'ont pas de bf16 efficace. L'état du `GradScaler` est dans le checkpoint.
  Non exercée ici (pas de GPU) : le chemin CUDA est écrit mais pas testé.
- **Suivi** : `metrics.csv` (colonnes `step, epoch, split, loss, policy_loss, value_loss, top1, top3, lr,
  pos_per_s, elapsed_s`). Lignes `train` : moyenne sur les `log_every` derniers pas, avec BatchNorm en mode
  entraînement. Lignes `val` : sous-ensemble fixe de 20 000 positions de validation (mode eval), tous les
  `eval_every` pas. Ligne `val_full` : toute la validation (60 902 positions) à la fin. top1 / top3 = le coup
  joué est le 1er choix / parmi les 3 premiers, classement parmi les coups légaux.
- **Checkpoints** : évaluation et sauvegarde coïncident, donc `best.pt` est exactement l'état de la
  meilleure perte de validation (sur le sous-ensemble). `last.pt` tous les `save_every` pas. Au redémarrage, les
  lignes du CSV postérieures au checkpoint repris sont supprimées. Reprise refusée si le réseau a changé.

### Tests (`tests/test_train.py`, 47 tests au total)

Mini jeu de données synthétique mais apprenable (640 + 96 positions de parties aléatoires ; cible de coup = le coup
légal d'indice minimal, cible de valeur = signe de la différence de matériel, 30 % de positions à
`value_valid = 0`), réseau 1 bloc × 16 filtres :
la perte politique est divisée par plus de 2 en 60 pas, la précision monte de plus de 20 points, la perte valeur
baisse, la validation (positions jamais vues) bat le hasard ; **une reprise après coupure donne exactement les mêmes
poids (`torch.equal`) et le même CSV qu'un run d'une traite** (coupure au pas 13, checkpoint au pas 10) ; lignes
CSV postérieures au checkpoint supprimées ; `best.pt` = meilleure validation ; un run terminé n'est pas
ré-entraîné ; reprise avec un autre réseau refusée ; masque depuis le format compact = masque de la vraie position
(Blancs et Noirs au trait) ; lots identiques avec et sans pool ; courbe de taux d'apprentissage ; `topk_correct`.
Mutations vérifiées : repartir du pas 0 à la reprise fait échouer le test de reprise ; oublier le masque dans
`topk_correct` fait échouer son test.

### Entraînement réel (tiny, 5 000 pas, CPU)

Commande : `scripts/train.py --config tiny --out checkpoints/tiny_sup`. 4 × 64, 352 460 paramètres, lots de 256,
5 000 pas = 1,28 M de positions = 0,43 époque (une époque = 11 496 pas). **784 s (13 min)**, ~1 800 positions/s
(1 440 au début, mesurées avec le réseau et les masques ; passer de 14 à 8 threads ne change rien).

| pas | perte pol. (val) | perte val. (val) | top-1 (val) | top-3 (val) |
|---|---|---|---|---|
| 500 | 2,512 | 0,869 | 28,2 % | 51,2 % |
| 1 000 | 2,374 | 0,852 | 31,5 % | 55,4 % |
| 2 000 | 2,232 | 0,817 | 34,5 % | 59,6 % |
| 3 000 | 2,145 | 0,809 | 35,9 % | 62,3 % |
| 4 000 | 2,087 | 0,792 | 37,2 % | 63,8 % |
| 5 000 | 2,069 | 0,784 | 37,8 % | 64,4 % |
| **5 000, toute la validation** | **2,062** | **0,784** | **38,2 %** | **64,5 %** |

- Référence pour la politique : au hasard parmi ~30 coups légaux, perte ≈ 3,4 et top-1 ≈ 3 %. Pour la valeur :
  prédire toujours 0 donne une MSE de 0,92 sur les positions valides ; le réseau fait 0,784 (−15 %), ce qui est
  modeste mais normal : le résultat d'une partie est une cible très bruitée à partir d'une seule position.
- Pas de surapprentissage (moins d'une demi-époque) : la validation suit l'entraînement ; la courbe baisse encore
  à la fin même avec le taux d'apprentissage au minimum, donc davantage de pas aideraient. `best.pt` = pas 5 000.
- Courbes : [docs/img/tiny_sup_curves.png](img/tiny_sup_curves.png).
- `show_policy.py` sur `best.pt` : position initiale → e4 50,8 %, d4 30,4 %, Nf3 6,7 %, c4 5,4 %, f4 1,0 %, valeur
  +0,014 ; après 1.e4 e5 2.Nf3 → Nc6 65,7 % ; sur `6k1/5ppp/8/8/8/8/5PPP/R5K1 w` (mat du couloir en un coup)
  Ra8# est le 1er choix avec 17,0 % (valeur +0,64) : le réseau a appris les ouvertures usuelles et un peu de
  tactique, mais reste faible (ce n'est qu'un réseau tiny, 13 minutes).
