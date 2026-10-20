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

## Étape 4b, partie 1 — GitHub, données 24 M, Dataset Kaggle (2026-10-20)

### Git et GitHub

- Adresse remplacée par `345045991+Enz091@users.noreply.github.com` dans la config locale, dans l'auteur et le
  committer des 7 commits existants (`git filter-branch`, historique réécrit avant toute publication) **et** dans
  le contenu de `CLAUDE.md` à tous les commits (l'ancienne adresse y figurait depuis l'étape 0). Vérifié après
  `reflog expire` + `gc --prune=now` : `git log -S` sur l'ancienne adresse ne trouve rien, aucune occurrence dans
  `.git`, dans les messages ni dans les fichiers suivis. Le **nom** d'auteur « Enzo Montani » n'a pas été touché.
- Dépôt distant `https://github.com/Enz091/Project_chess.git`, `main` poussée (le gestionnaire d'identifiants
  Windows avait déjà une session GitHub : aucune connexion demandée).

### Extraction de 24 M de positions

- `data.extract` accepte maintenant une **liste de sources** (`--months AAAA-MM ...` dans `prepare_data.py`) et un
  quota `per_source_positions` par source. Numéros de partie = `indice_source * 100 000 000 + rang dans la source`
  (uniques, et indépendants du nombre de workers : un premier essai basé sur un compteur global donnait des
  numéros différents selon les workers à cause des lots lus d'avance, le test l'a détecté). Le manifeste contient
  le détail par source. Un test de plus (`test_several_sources_with_quota`) : 48 tests au total.
- Commande : `prepare_data.py --months 2026-02 ... 2026-09 --per-source-positions 3000000 --out data/lichess_24m
  --workers 7` (7 workers + le processus de lecture = 8 processus). Mêmes filtres qu'à l'étape 3 (Elo ≥ 2000 des
  deux joueurs, pas de bullet, pas de BOT, `value_valid = 0` pour les Time forfeit).
- Résultat : **24 035 177 positions** (train 23 560 129, val 475 048), 311 517 parties gardées sur 5 364 843 lues
  (4,45 M refusées pour l'Elo, 581 k bullet, 5 057 BOT), 1,80 Go sur disque, 1,74 Go téléchargés, **317 s**.
  `value_valid = 1` pour 71 % des positions d'entraînement ; résultats équilibrés (+1 : 10,83 M, −1 : 10,70 M,
  nulles : 2,03 M) ; aucune partie dans train ET val ; le mois 2026-09 redonne exactement les 3 004 035 positions de
  l'étape 3 (déterminisme).
- **Limite de la diversité** : le flux zstd ne permet pas de sauter dans le fichier, donc chaque mois ne
  contribue que par ses premières ~650 000 parties (soit les premières heures du 1er du mois, ~0,5 % du mois).
  La diversité vient des 8 mois différents (autres parties, autres adversaires, évolution des joueurs), pas
  de la répartition dans le mois. Lire une journée entière coûterait plusieurs Go de téléchargement par mois.

### Dataset Kaggle

- `scripts/make_kaggle_zip.py data/lichess_24m --out data/kaggle/lichess_24m.zip` : 98 shards + `manifest.json`
  à la racine de l'archive. Les shards se compressent ~7x : **256 Mo** à téléverser pour 1,80 Go de données.
  Kaggle décompresse le `.zip` à la création du dataset, donc les shards seront directement dans
  `/kaggle/input/<nom-du-dataset>/` (donc `load_split("/kaggle/input/<nom>", "train")`). RAM une fois chargé :
  ~1,8 Go pour le train.

### Les masques de coups légaux ralentiront-ils le GPU ? (mesures et estimation, rien n'est changé)

- Mesuré sur ce PC (processeur au repos) : `legal_masks_from_packed` = **9 400 positions/s par cœur**
  (30,1 coups légaux en moyenne), 25 500 pos/s avec un pool de 3 processus (retour des masques inclus),
  `unpack_batch` 390 000 pos/s, envoi d'un lot de 1024 masques entre processus (4,8 Mo) : ~7 ms.
- Besoin du GPU (estimation, pas de GPU ici) : le réseau kaggle fait 200 M multiplications-additions par position,
  soit ~1,2 GFLOP pour un pas d'entraînement complet. Sur T4 en fp16 on peut raisonnablement attendre
  **6 000 à 13 000 pos/s**, sur P100 **4 000 à 8 000**. À 10 000 pos/s, une époque de 23,6 M de positions
  dure ~40 min.
- Kaggle a 4 cœurs : 1 pour le processus d'entraînement (lancement des kernels CUDA, assemblage des lots) et au
  mieux 3 workers de masques. Ces cœurs sont probablement plus lents que ceux de ce PC (hypothèse : 0,5 à
  0,8x), soit 3 × 5 000 à 7 000 = **15 000 à 21 000 pos/s au mieux, 10 000 à 12 000 si les cœurs sont lents ou
  contendus**. Conclusion : avec ce réseau et un T4/P100, le CPU devrait **tout juste suivre** ; ce n'est pas
  catastrophique mais il n'y a aucune marge, et le CPU devient le goulot si le GPU est rapide ou si on utilise
  les 2 T4. Le risque réel est de perdre 0 à 40 % du débit selon la machine tirée, sans pouvoir le savoir
  avant d'avoir lancé. Une mesure de 2 minutes sur Kaggle (pos/s avec masques en ligne vs sans masque) lèverait
  le doute.

## Étape 4b, partie 2 — Mesures de débit, perte sans masque, notebook Kaggle (2026-10-20)

Décision de l'auteur : **mesurer d'abord** sur Kaggle ; pas de précalcul des masques pour l'instant.

### Diagnostic de débit (`train.py`)

- Le temps d'un pas est coupé en **attente des données** (obtenir le lot suivant : indexation, `unpack_batch`, attente
  des masques calculés par le pool) et **calcul** (passe avant/arrière, métriques). La boucle est synchrone (les
  `.item()` de la perte attendent le GPU), donc les deux temps s'additionnent honnêtement : un GPU affamé se voit
  comme une attente élevée. `PhaseTimer` cumule data / masques / calcul par fenêtre de `log_every` pas ;
  `GpuSampler` échantillonne l'utilisation du GPU toutes les 0,5 s dans un fil (pynvml, sinon `nvidia-smi`,
  sinon « n/d » sur CPU). Taux NVML = part du temps où un noyau CUDA tournait : bas = le GPU attend, mais 100 %
  ne prouve pas la saturation.
- Nouvelles colonnes de `metrics.csv` : `gpu_util`, `data_wait_pct`, `mask_wait_pct` (vides si non mesurées).
  `pos_per_s` exclut maintenant la validation et les sauvegardes (avant, la fenêtre suivant une validation était
  sous-estimée).
- Ligne de log : `... 1 234 pos/s | GPU 93% | attente données 4% (dont masques 2%)`.
- `scripts/summarize_run.py <dossier>` : débit moyen/min/max, GPU, attente, durée d'une époque, **verdict**
  (le GPU « attend » si attente > 15 % ou GPU < 80 % ; il dit si les masques en sont la cause principale).
- `torch.backends.cudnn.benchmark = True` sur CUDA (formes fixes). Le chemin CUDA reste non testé faute de GPU.

### Budget de temps : `--max-minutes`

`train(..., max_minutes=)` / `scripts/train.py --max-minutes N` : après N minutes d'entraînement (chargement des
données exclu), sauvegarde `last.pt` **au pas courant** (même hors `save_every`) et s'arrête (`timed_out`). Relancer
reprend. Sert à l'essai de 10 minutes et à finir une session Kaggle avant la coupure des 12 h. Testé : arrêt au pas 1
puis reprise = mêmes poids (`torch.equal`) qu'un run d'une traite.

### Option perte sans masque (désactivée par défaut)

- `[train] mask_train_loss = true` (tiny.toml et kaggle.toml) ; `false` ou `--no-mask-loss` : softmax sur les 4 672
  sorties pour la perte d'entraînement (`policy_value_loss(..., legal_mask=None)`), `Batch.legal = None`, le pool de
  masques n'est plus sollicité à l'entraînement. Les top-1/top-3 des fenêtres d'entraînement sont alors calculés
  sans masque.
- **Gardés avec masque** : la validation (`ValSet`, sous-ensemble et validation complète : perte masquée, top-k parmi
  les coups légaux, donc comparable aux résultats de l'étape 4a) ; et tout ce qui sert au jeu (`masked_policy`,
  MCTS). Les pertes « train » d'un run sans masque ne sont **pas** comparables à celles d'un run masqué (la
  perte sans masque est un peu plus haute : de la probabilité reste sur les coups illégaux au début).
- À décider après la mesure : si le GPU attend les masques → `mask_train_loss = false` dans `kaggle.toml`.
  À surveiller alors : que le top-1 de validation (masqué) ne soit pas plus mauvais à pas égal.
- Tests (56 au total) : perte sans masque = `cross_entropy` standard, top-k sans masque, lots sans masque =
  mêmes entrées avec `legal = None`, le chronomètre ne compte les masques que s'ils sont calculés, `GpuSampler`
  (lecture injectée), un entraînement sans masque apprend et valide avec masque, colonnes de diagnostic, budget de temps.

### Notebook `notebooks/kaggle_train.ipynb`

Cellules : 0 réglages (`RUN_BENCH`, `RUN_FULL`, `NO_MASK_LOSS`, durées...) ;
1 clone + `pip install -e .` sans torch (vérifie que la version de torch n'a pas bougé) ; 2 environnement (Python
≥ 3.11, torch, cœurs CPU, RAM, disque, GPU, test fp16 sous autocast, `nvidia-smi`) ; 3 recherche du Dataset
(`**/train_00000.npz` sous `/kaggle/input`) ; 4 reprise ; 5 essai 10 min avec masques + résumé ; 6 essai 4 min sans
masque + comparaison + effacement des `.pt` d'essai ; 7 entraînement complet (`RUN_FULL`).
Exécuté en local sur un faux `/kaggle/input` (config tiny, CPU, `ALLOW_CPU`) : toutes les cellules passent ; la
cellule de reprise choisit bien le `kaggle_sup/last.pt` du pas le plus avancé et ignore les `bench/last.pt`.
**Jamais exécuté sur Kaggle** (GPU, pynvml, chemins réels non vérifiés).

### Reprise entre deux sessions Kaggle

`/kaggle/working` disparaît à la fin d'une session ; seule la **sortie d'une version sauvegardée** (Save Version →
*Save & Run All (Commit)*) est conservée. Chaînage : dans la nouvelle session, *Add Input* → ce même notebook (sa
dernière sortie) → la cellule 4 copie `kaggle_sup/{last,best}.pt` et `metrics.csv` dans `/kaggle/working/kaggle_sup`
et l'entraînement reprend au pas sauvegardé. Repli manuel : télécharger `last.pt` depuis l'onglet Output, le déposer
dans un petit Dataset privé en gardant le dossier `kaggle_sup/`. `FULL_MINUTES = 660` fait finir proprement la session
(sinon une coupure à 12 h peut faire échouer la version et perdre la sortie).
- `kaggle.toml` : `total_steps = 20000` valait ~7 époques de 2,9 M de positions ; avec 23,56 M de positions une
  époque = 23 008 pas, donc 20 000 pas < 1 époque. À revoir après la mesure de débit.
