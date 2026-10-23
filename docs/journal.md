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

## Étape 4b, partie 3 — Résultats de l'essai Kaggle et réglage de l'entraînement (2026-10-20)

### Essais de débit sur Kaggle (config kaggle, 10 × 128, lots de 1024, fp16)

| essai | débit | GPU | attente des masques |
|---|---|---|---|
| **avec masques** (10 min) | **7 970 positions/s** | **94 %** | **0 %** |
| sans masque (`--no-mask-loss`) | 8 016 positions/s (**+0,6 %**) | — | — |

- Le GPU ne manque pas de données : les 3 workers de masques précalculent assez de lots à l'avance, l'attente est nulle
  et le GPU est occupé à 94 %. Le débit mesuré (≈ 8 000 pos/s) est dans la fourchette estimée au journal (6 000 à
  13 000 sur T4). Une époque de 23,56 M de positions dure ≈ 49 min.
- L'essai sans masque n'apporte que +0,6 % de débit **et apprend plus lentement** (observation de l'auteur) : on
  **garde les masques** (`mask_train_loss = true`). L'option reste dans le code, désactivée. Le précalcul des masques
  n'a pas lieu d'être.
- **Précision au bout des 10 minutes** (avec masques, 4 580 pas = 4,7 M de positions = 0,20 époque, validation sur le
  sous-ensemble fixe de 30 000 positions, au dernier point de validation, pas 4 000) : perte totale **2,593**
  (politique **1,759**, valeur **0,835**), **top-1 44,7 %**, **top-3 72,8 %**. En fin d'essai, sur la fenêtre
  d'entraînement : top-1 45,3 %, top-3 73,8 %. Pour situer : le réseau tiny de l'étape 4a, après 5 000 pas de 256
  (0,43 époque, 1,3 M de positions), était à 38,2 % / 64,5 % (politique 2,062 sur toute la validation) ; le
  réseau kaggle fait déjà nettement mieux (+6,5 points de top-1, +8 de top-3) avec 0,20 époque, mais les deux
  mesures ne sont pas sur les mêmes données de validation (autres mois, autre taille).
- **Bruit de la perte de valeur** : 0,781 au pas 3 000 puis 0,835 au pas 4 000. Une telle remontée en 1 000 pas
  d'un réseau qui apprend vient très probablement du bruit de mesure (30 000 positions, `value_valid = 1`
  seulement, issues de peu de parties : les positions d'une même partie partagent le même résultat, donc le nombre
  effectif d'exemples est bien plus petit que 30 000) et non d'une dégradation. C'est la raison du passage à
  100 000 positions de validation. À surveiller sur le vrai entraînement : si la perte de valeur de validation
  remonte durablement avec 100 000 positions, ce ne sera plus du bruit.
- **Essai sans masque** (validation au pas 1 000, 30 000 positions) : top-1 **35,6 %**, top-3 **61,7 %**, contre
  **36,3 %** et **62,6 %** avec masques au même pas, soit −0,7 point de top-1 et −0,9 point de top-3. Un seul
  point de mesure, une seule graine : c'est un indice cohérent avec « apprend plus lentement », pas une
  preuve ; combiné au gain de débit de seulement 0,6 %, cela suffit pour garder les masques.

### Réglages de `configs/kaggle.toml` pour l'entraînement complet

- `total_steps = 92000` : 23 008 pas par époque (23 560 129 / 1 024), soit **4,00 époques** (94,2 M de positions) ;
  à 7 970 pos/s : 3,28 h d'entraînement pur, ~3 h 30 avec validations et sauvegardes (< `FULL_MINUTES` = 660 : une
  seule session suffit).
- `val_positions = 100000` (au lieu de 30 000) : sous-ensemble fixe plus grand (tiré au hasard dans les 475 048
  positions de validation, donc issu de bien plus de parties), parce que la perte de valeur — calculée sur
  les seules positions `value_valid = 1`, et corrélée au sein d'une même partie — était trop bruitée avec 30 000
  positions de peu de parties. Coût : les masques de ces 100 000 positions sont calculés une fois au démarrage
  (~4 s à 25 000 pos/s avec 3 workers, ~470 Mo de RAM) et chaque validation (tous les 1 000 pas, 92 fois) dure quelques
  secondes de plus.
- **Warmup et cosinus vérifiés** avec `lr_factor` sur la config réelle : `warmup_steps = 1000` (1,1 % du total) ; le taux
  monte linéairement de 2e-6 (pas 0) à 2e-3 (pas 999, 1,0 du maximum), puis décroît en cosinus, monotone, passe par
  1,0e-3 vers le pas 47 000 (milieu) et finit à 2,0e-5 (= 1 % du maximum) au dernier pas. Le cosinus est calculé
  sur `total_steps - warmup_steps`, donc il s'étire tout seul sur 92 000 pas. `eval_every = 1000` et `save_every = 500`
  restent des multiples de `log_every = 100` et divisent 92 000. Le calendrier dépend de `total_steps` : ne plus
  le changer en cours de route, sinon une reprise suivrait une autre courbe.

### Notebook : corrections

- **`ModuleNotFoundError: chesszero` (cellule de comparaison)** : le paquet est installé par `pip install -e` alors que le
  noyau tourne déjà, il n'est donc visible que des sous-processus (`scripts/train.py`). Correction : la cellule 1
  fait `sys.path.insert(0, REPO_DIR/"src")` + `importlib.invalidate_caches()` juste après l'installation (aussi quand
  `MANAGE_REPO = False`). Une seule cellule importe `chesszero` dans le noyau (l'import de `read_metrics` de la cellule
  6) ; les autres n'importent que `torch` et la bibliothèque standard. Vérifié en local en cachant le paquet au noyau
  (finder d'installation éditable et chemin `src` retirés, import qui échoue avant la cellule 1) : toutes les cellules,
  dont la comparaison, passent.
- **Valeurs par défaut de la cellule 0** : `RUN_BENCH = False`, `NOMASK_MINUTES = 0`, `RUN_FULL = True`
  (92 000 pas, `FULL_MINUTES = 660`), `NO_MASK_LOSS = False`. Le notebook lance donc directement l'entraînement complet
  (après la reprise éventuelle de la cellule 4).

## Étape 4b, partie 4 — Résultat de l'entraînement complet sur Kaggle (2026-10-20)

### Fichiers récupérés et rangés

- Téléchargements : `best.zip`, `last.zip` (37 842 913 octets chacun) et `metrics.csv` (152 138 octets, 1 014 lignes).
  Les deux « .zip » sont en fait des checkpoints PyTorch (le format `torch.save` est lui-même une archive zip :
  les entrées s'appellent `ckpt_00092000.pt/...`), renommés par le navigateur. **Ils sont identiques octet pour octet**
  (même SHA-256) : normal, au dernier pas la validation était la meilleure, donc `best.pt` est une copie de `last.pt`.
  Aucun dossier extrait n'existait dans Téléchargements.
- Copiés (sans extraction) en `checkpoints/kaggle_sup/best.pt`, `last.pt`, `metrics.csv` ; comparaison octet à octet
  faite, puis les trois originaux de Téléchargements supprimés.
- **Vérification** : les deux se chargent avec `torch.load` et `build_model(ModelConfig.named("kaggle"))`
  (`load_state_dict` strict, 3 140 940 paramètres) ; `step = 92000` (= `total_steps`), `epoch = 3` (compteur entier,
  4,00 époques réelles), `config.model == kaggle` et `config.train == [train] de kaggle.toml` (dont `val_positions = 100000`,
  `mask_train_loss = true`), `best_val_loss = 2,1387`, temps écoulé 3,77 h.

### Courbes et chiffres de validation

Courbes : [docs/img/kaggle_sup_curves.png](img/kaggle_sup_curves.png) (`plot_curves.py checkpoints/kaggle_sup --out ...`).
Validation = sous-ensemble fixe de 100 000 positions, toutes les 1 000 pas (92 points) ; fin = toute la validation
(475 048 positions, 330 934 avec `value_valid = 1`).

| pas | époque | perte | politique | valeur | top-1 | top-3 |
|---|---|---|---|---|---|---|
| 1 000 | 0,04 | 2,966 | 2,139 | 0,827 | 36,0 % | 62,6 % |
| 5 000 | 0,22 | 2,498 | 1,731 | 0,768 | 45,5 % | 73,8 % |
| 10 000 | 0,43 | 2,359 | 1,611 | 0,748 | 48,2 % | 76,9 % |
| 23 000 | 1,00 | 2,261 | 1,523 | 0,737 | 50,6 % | 79,0 % |
| 46 000 | 2,00 | 2,193 | 1,459 | 0,734 | 52,3 % | 80,6 % |
| 69 000 | 3,00 | 2,150 | 1,421 | 0,729 | 53,0 % | 81,5 % |
| 92 000 (sous-ensemble) | 4,00 | 2,139 | 1,410 | 0,728 | 53,6 % | 81,7 % |
| **92 000, validation complète** | 4,00 | **2,135** | **1,407** | **0,728** | **53,45 %** | **81,85 %** |

- Pour situer : tiny (étape 4a, 0,43 époque) : 2,062 / 0,784 / 38,2 % / 64,5 % sur sa propre validation de 60 902
  positions ; le réseau kaggle gagne +15 points de top-1 et +17 de top-3 (les deux validations ne sont pas les mêmes
  données). Valeur : toujours prédire 0 donne 0,899 sur la validation complète ; le réseau fait 0,728 (−19 %).
- Gains par époque (moyenne des validations de l'époque) : politique 1,649 → 1,489 → 1,440 → 1,414 ; valeur 0,762 → 0,740
  → 0,731 → 0,729 ; la dernière époque (taux d'apprentissage de 3e-4 à 2e-5) apporte encore ~0,025 nat de politique
  (+0,6 point de top-1) mais presque rien sur la valeur.
- Débit : médiane 7 275 pos/s sur tout le run (7 970 pendant l'essai), GPU médian 95 %, 13 640 s au total
  (3,79 h, dont ~8 min avant le pas 100 : chargement des 98 shards et masques de validation).

### La perte valeur de validation a-t-elle remonté ? Non, pas durablement

- Minimum **0,7266 au pas 65 000**, valeur finale 0,7284 (+0,002) ; sur la dernière époque elle reste entre 0,7273 et
  0,7300. Pente (régression linéaire des points de validation) : −0,00001 par 1 000 pas depuis le pas 62 000,
  −0,00013 depuis le pas 46 000 : **plateau**, pas de remontée. Sur 91 transitions, 43 sont des hausses : c'est du bruit
  autour d'une tendance baissière ; la plus grosse (+0,040 au pas 4 000, le 0,828 de l'essai de 10 min) est dans la
  phase de bruit du début, et les points sont devenus bien plus stables avec 100 000 positions.
- **Mais un début de surapprentissage de la tête valeur** : écart entraînement/validation de la perte valeur (moyenne par
  époque) 0,012 → 0,018 → 0,025 → 0,040 (train 0,750 → 0,722 → 0,706 → 0,689 contre val 0,762 → 0,740 → 0,731 → 0,729).
  La courbe d'entraînement fait des marches vers les pas 46 000 et 69 000 (début des époques 3 et 4 : le réseau
  retrouve des positions déjà vues et les reconnaît). Le résultat d'une partie est une cible bruitée et partagée par
  toutes les positions d'une même partie, donc le réseau finit par mémoriser les parties plutôt que par généraliser.
  Politique : écart plus faible (train 1,357 contre val 1,410 en fin de run) et la validation baisse encore.
- Conséquence : des époques supplémentaires sur ces mêmes données n'aideraient presque que la politique ; pour la
  valeur, il faut d'autres positions (self-play, plus de parties) plutôt que plus de passes.

### `show_policy` : réseau kaggle (pas 92 000) contre tiny (pas 5 000)

| position | tiny | kaggle |
|---|---|---|
| initiale | e4 50,8 %, d4 30,4 %, Nf3 6,7 % ; valeur +0,014 | e4 51,9 %, d4 30,4 %, Nf3 6,3 % ; +0,039 |
| 1.e4 e5 2.Nf3 (Noirs) | Nc6 65,7 % | Nc6 71,6 %, Nf6 12,0 % ; valeur −0,030 |
| mat du couloir `6k1/5ppp/8/8/8/8/5PPP/R5K1 w` | **Ra8# 17,0 %** (1er), Ra7 15,0 % ; valeur +0,64 | Ra7 17,9 % (1er), **Ra8# 17,2 %** ; valeur +0,93 |
| D+R contre R, mat en 1 `7k/5Q2/6K1/8/8/8/8/8 w` (Qg7#, Qh7#, Qf8#, Qe8# mattent tous) | Qf8# 29,9 %, Qe8# 9,0 %, Qh7# 5,8 % (Qg7# hors du top 5) ; valeur +0,93 | **Qg7# 44,9 %, Qh7# 41,5 %, Qf8# 7,9 %, Qe8# 1,9 %** (96 % sur des mats) ; +0,97 |
| mat du berger `r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w` | Qxe5+ 15,2 %, Qf3 11,4 %… (**rate le mat**) ; valeur −0,02 | **Qxf7# 79,8 %**, Bxf7+ 17,3 % ; valeur +0,21 |
| dame noire en prise `rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w` | Nxh4 46,9 % ; valeur 0,00 | **Nxh4 73,4 %** ; valeur +0,41 |
| finale roi + pion `8/8/4k3/8/4PK2/8/8/8 w` | e5 44,9 %, Ke3 40,4 % ; valeur +0,52 | e5 91,7 % ; valeur +0,26 |

- Le réseau kaggle est nettement plus net sur la tactique d'un coup (capturer une dame en prise, mat du berger, mat à
  la dame : 96 % de la probabilité sur des coups qui matent) et plus confiant sur les ouvertures courantes. La valeur
  est mieux calibrée (+0,93 au mat du couloir, +0,41 avec une dame d'avance).
- **Pas de progrès sur le mat du couloir** : Ra7 passe devant Ra8# (17,9 % contre 17,2 %), les deux restent faibles. Cette
  position (aucune pièce noire hormis 3 pions) est très éloignée des parties réelles ; le réseau imite des coups
  humains et ne calcule rien. C'est l'illustration de ce que le MCTS apportera (la recherche trouvera le mat).
- Finale roi + pion : e5 à 91,7 % montre une forte préférence, mais **je n'ai pas vérifié** que c'est le bon coup ni
  que la position est gagnante (pas de moteur ni de tables de finales ici) ; ne pas s'en servir comme preuve.

## Étape 5 — MCTS (2026-10-20)

### Ce qui a été écrit

- **`src/chesszero/mcts.py`** :
  - `MCTSConfig` (section `[mcts]` des configs) ;
  - `terminal_value(board)` ;
  - `NetEvaluator` : un appel au réseau par lot, softmax sur les seuls indices des coups légaux ;
  - `policy_best_move` : meilleur coup du réseau sans recherche ;
  - `Node`, `MCTS` : `search`, `best_move`, `visit_distribution`, `advance`, `root_stats`, `principal_variation`, `check_tree`.
- **Un nœud par position atteinte.** Les statistiques de ses arêtes sont des tableaux numpy (`P`, `P_raw`, `N`, `W`, `VL`), donc le score PUCT d'un nœud se calcule en une opération. Les fils sont créés quand l'arête est choisie pour la première fois.
- **Les nœuds ne stockent pas de position.** L'objet `MCTS` garde un seul `chess.Board` avec tout l'historique, joue les coups pendant la descente (`push`) et revient en arrière avec `pop`. La feuille est copiée sans pile pour l'évaluation.
- **PUCT** : `Q + c_puct · P · √(1 + ΣN + ΣVL) / (1 + N + VL)`. Le `1 +` sous la racine compte l'évaluation du nœud lui-même, comme AlphaZero : sans lui, la première sélection après l'expansion aurait un terme d'exploration nul pour tous les coups.
- **Conventions de signe** :
  - la valeur du réseau est vue par le camp au trait dans la position évaluée ;
  - `W` et `Q` d'une arête sont vus par le camp qui **choisit** ce coup, ce qui permet de prendre directement l'argmax ;
  - la remontée change le signe **avant** d'ajouter la valeur à l'arête qui mène à la feuille (elle appartient à l'adversaire), puis à chaque niveau.
- **FPU (coups jamais visités)** : `Q = node.q() − fpu_reduction`, avec `node.q() = (valeur réseau du nœud + ΣW) / (1 + ΣN)`.
  - C'est la valeur moyenne du nœud **pour le camp qui choisit dans ce nœud**.
  - Elle est égale à l'opposé du Q de l'arête entrante (vérifié dans `check_tree`), mais elle est aussi définie à la racine, y compris après réutilisation de l'arbre. Au départ, c'est la valeur du réseau pour ce nœud.
  - On ne compte pas la perte virtuelle dans cette valeur de base.
- **Positions terminales** (`terminal_value`, calculée une fois à la création du nœud, avec l'historique) :
  - mat → −1 ;
  - pat, matériel insuffisant, 50 coups (`halfmove_clock ≥ 100`, nulle réclamable) → 0 ;
  - **triple répétition exacte** (`is_repetition(3)`) → 0. Le test n'est fait que si `halfmove_clock ≥ 8`, car trois occurrences demandent au moins 8 demi-coups réversibles.
  - La valeur reste valable après `advance`, car le nouvel historique prolonge celui du nœud.
- **Évaluation par lots** :
  - Les descentes posent une perte virtuelle : `VL += 1` sur chaque arête. Une arête en attente paraît perdue (`Q = (W − vl·VL)/(N + VL)`).
  - Une feuille terminale est remontée tout de suite, sans réseau.
  - Une **collision** (feuille déjà en attente dans le lot) annule la descente sans compter de simulation ; le lot se ferme après `batch_size` collisions.
  - `VL` est un compteur entier séparé de `N`, ce qui permet de vérifier qu'il revient exactement à 0.
- **`search(N)` ajoute exactement N simulations** (le dernier lot est tronqué). Une simulation = une feuille évaluée ou terminale remontée. La première évaluation de la racine n'en est pas une.
- **Réutilisation de l'arbre** : `advance(move)` fait du fils la nouvelle racine, pour nos coups comme pour ceux de l'adversaire ; le reste est libéré. Une arête comptait aussi l'évaluation de son fils, donc la nouvelle racine a `N(arête) − 1` visites.
- **Self-play, prévu mais désactivé par défaut** :
  - bruit de Dirichlet à la racine à chaque `search`, recalculé depuis `P_raw` (il ne s'accumule pas quand un nœud redevient racine) ;
  - température dans `best_move` (`T = 0` : max des visites, égalité départagée par le prior ; `T > 0` : tirage selon `N^(1/T)`) ;
  - `visit_distribution()` donne le vecteur (4672,) qui servira de cible de politique.
- **`checkpoint.load_for_inference(path, device)`** : reconstruit le réseau depuis la config du checkpoint. `show_policy.py` l'utilise.
- **Configs, section `[mcts]`** (identique dans tiny et kaggle) : `num_simulations = 200`, `c_puct = 1.5`, `batch_size = 8`, `virtual_loss = 1`, `fpu_reduction = 0.2`, `dirichlet_alpha = 0.3`, `dirichlet_epsilon = 0` (désactivé), `temperature = 0`.
- **Scripts** :
  - `show_search.py` : prior, visites, Q et variante principale d'une position ;
  - `bench_mcts.py` : simulations/s selon la taille de lot ;
  - `match.py` : MCTS contre politique seule, PGN dans `docs/`.

### Tests (`tests/test_mcts.py`, 46 tests ; 102 au total)

- **Évaluateurs factices** (priors uniformes ; valeur nulle, constante, différence de matériel, ou « Blancs +0,5 »).
- **Config** : chargement de `[mcts]` (bruit et température désactivés) ; refus d'une clé inconnue.
- **Positions terminales** :
  - mat, pat, roi contre roi, roi + fou contre roi, 50 coups (99 → pas encore, 100 → nulle) ;
  - triple répétition (Nf3 Nf6 Ng1 Ng8 ×2 : nulle au 8e demi-coup, pas avant) ;
  - recherche refusée sur une position terminale ;
  - dans l'arbre, la répétition est détectée avec l'historique de la partie : l'arête vaut exactement 0 alors que le réseau factice dit +0,9.
- **Comptage** : la somme des visites de la racine vaut exactement le nombre de simulations (lots de 1, 8, 16, et 64 avec 50 simulations), après une deuxième recherche, et après `advance` (`N(arête) − 1` puis `+ 50`).
- **Perte virtuelle** :
  - après **chaque** lot (`_run_batch` instrumenté), tous les `VL` valent 0 et aucun nœud n'est en attente ;
  - `check_tree` détecte une perte virtuelle oubliée ;
  - un lot de 16 trouve 16 feuilles distinctes sans collision.
- **Signes** :
  - une simulation avec une feuille à +0,5 donne −0,5 sur l'arête de la racine ;
  - avec « Blancs +0,5 » partout, **toutes** les arêtes valent exactement +0,5 quand les Blancs choisissent et −0,5 quand ce sont les Noirs, à toutes les profondeurs, depuis la position initiale et depuis sa version retournée ;
  - mat en un trouvé avec des priors uniformes, pour les Blancs et pour les Noirs ;
  - dame en prise capturée, pour les deux camps.
- **FPU** (dame d'avance, Blancs puis Noirs au trait) :
  - à la racine, la FPU des coups inexplorés est **> +0,5** (≈ q − 0,2) ;
  - au niveau suivant, où le camp qui perd choisit, elle est **< −0,5** ;
  - la base vaut −Q de l'arête entrante.
- **Bruit, température, réutilisation** :
  - bruit absent par défaut (`P is P_raw`) ;
  - avec ε = 0,25, priors modifiés, somme 1, reproductibles avec la même graine, recalculés depuis `P_raw` après `advance` ;
  - T = 0 → coup le plus visité ; T = 1 → fréquences de tirage ≈ visites (3 000 tirages, écart < 0,03) ;
  - `visit_distribution` est indexée comme `move_to_index` et somme à 1 ;
  - le sous-arbre est conservé (même objet) ; un coup inexploré donne une racine vide ; un coup illégal est refusé.
- **Vrai réseau** (sautés si `checkpoints/kaggle_sup/best.pt` est absent), lots de 1, 8 et 16, 400 simulations, arbre vérifié après chaque lot : le MCTS trouve les mats ci-dessous, et la politique seule choisit bien l'autre coup.
- **Tests de mutation (faits à la main, code restauré)** :
  - FPU de signe inversé (`−q − 0,2`) : les 2 tests FPU échouent ;
  - remontée sans changement de signe : 17 tests échouent (signes, mats, dame en prise, mats du réseau).

### Mats : politique seule contre recherche (réseau kaggle, lots de 8)

| position | politique seule | MCTS 200 simulations | MCTS 400 simulations |
|---|---|---|---|
| mat du couloir `6k1/5ppp/8/8/8/8/5PPP/R5K1 w` | **Ra7** 17,9 % (Ra8# 17,2 %) | **Ra8#** 64 visites (32 %), Q = +1,000 ; Kf1 25, Ra7 25 | **Ra8#** 136 (34 %) ; Ra7 50 (Q +0,965) |
| même position retournée, Noirs au trait | Ra2 | **Ra1#**, mêmes chiffres exactement | **Ra1#**, mêmes chiffres |
| piège de Légal (mat en 2) `rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P3/2N5/PPPP1PPP/R1BbK2R w` (après 5...Bxd1) | **Nxf7** 59,2 % (Bxf7+ 40,1 %) ; valeur −0,16 | **Bxf7+** 189 (94,5 %), Q +0,92 ; Nxf7 Q −0,12 ; PV 6.Bxf7+ Ke7 7.Nd5# | **Bxf7+** 385 (96 %), Q +0,95 |
| mat étouffé de Philidor (mat en 2) `1r5k/6pp/7N/3Q4/8/8/8/K7 w` | **Nf7+** 68,4 % (Qg8+ 25,2 %) | **Qg8+** 97 (48,5 %) contre Nf7+ 88 ; PV 1.Qg8+ Rxg8 2.Nf7# | **Qg8+** 232 (58 %) contre Nf7+ 153 |

- Les deux mats en deux sont des positions dont j'ai vérifié par recherche exhaustive (python-chess) que le mat en deux est forcé et que le premier coup est le seul qui le force (aucun mat en un).
- La position retournée donne exactement les mêmes visites : l'encodage « du point de vue du camp au trait » rend les deux camps identiques pour le réseau, et le signe de la remontée est le bon pour les Noirs.
- **Pourquoi Ra8# n'a que ~1/3 des visites** : le réseau juge déjà la position gagnée (+0,93) et les autres coups gardent un Q de 0,95 à 0,97. La différence avec le mat (1,0) est petite à côté du terme d'exploration, donc les visites se répartissent. Le MCTS ne cherche pas le gain *le plus court* : gagner maintenant ou plus tard vaut +1 dans les deux cas. C'est sans importance ici, mais à garder en tête (allongement des finales gagnées, voir le match).
- **Philidor, serré à 200 simulations** (97 contre 88) : Nf7+ Kg8 Nh6+ est un échec perpétuel, et le réseau croit de toute façon la position gagnée (+0,96, dame contre tour). Avec 400 simulations l'écart est net. Le Q de Qg8+ (0,95) reste sous 1, car la moyenne contient encore les premières évaluations du réseau, faites avant que le mat soit découvert.

### Vitesse (CPU Intel, 14 threads, torch 2.14.1+cpu, réseau kaggle, 800 simulations, meilleur de 2)

| lot | simulations/s (position initiale) | simulations/s (milieu de jeu) | positions/appel | part du temps dans l'évaluation | coup (milieu de jeu) |
|---|---|---|---|---|---|
| 1 | 220 | 222 | 1,0 | 88-89 % | Ne5 |
| 2 | 234 | 234 | 2,0 | 90-91 % | Ne5 |
| 4 | 316 | 304 | 4,0 | 86-87 % | Ne5 |
| **8** | **432** | **426** | 7,9 | 83-84 % | **Ne5** |
| 16 | 546 | 531 | 15,4 | 81-83 % | Qe2 |
| 32 | 611 | 622 | 29,7 | 81-82 % | Qe2 |
| 64 | 680 | 658 | 50-53 | 81-83 % | Re1 |

- Le réseau prend plus de 80 % du temps à toutes les tailles. Python (descente, coups légaux, remontée) n'est pas le goulot.
- **Qualité selon la taille de lot** (milieu de jeu `r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w`, 800 simulations) :
  - lot 1 : Ne5 303, Qe2 247, Re1 108 ;
  - lot 8 : Ne5 282, Qe2 243, Re1 120 (même répartition) ;
  - lot 16 : Qe2 247, Ne5 240 (léger aplatissement, le coup change) ;
  - lot 64 : Re1 171, Qe2 171, Ne5 100. Les visites retombent vers les priors (Re1 = 31 % de prior, c'est le coup de la politique) : avec ~13 tours de 64 feuilles, la perte virtuelle disperse la recherche.
- **Recommandation, et valeur des configs : `batch_size = 8`** sur CPU, soit ~430 simulations/s (×2 par rapport aux lots de 1), sans perte visible de qualité. 16 est acceptable pour de gros budgets (≥ 800 simulations). Sur GPU (étape 6), les gros lots devront venir de **plusieurs parties jouées en parallèle**, pas d'une perte virtuelle plus forte dans un seul arbre.

### Match : MCTS-200 contre politique seule (`scripts/match.py`, 6 min)

- 10 ouvertures fixes de 6 demi-coups (italienne, sicilienne, française, Caro-Kann, gambit dame refusé, est-indienne, anglaise, espagnole, scandinave, London), chacune jouée deux fois avec les couleurs inversées. Réseau kaggle des deux côtés, lots de 8, réutilisation de l'arbre, limite de 400 demi-coups.
- **Résultat : 20 victoires du MCTS sur 20, toutes par mat** (10 avec les Blancs, 10 avec les Noirs) ; 0,45 s par coup MCTS (~440 simulations/s).
- **Écart Elo estimé : ≥ +317 (borne à 95 %)**. Avec 20 gains sur 20, la probabilité de gain p vérifie p ≥ 0,05^(1/20) = 0,86, soit au moins +317 Elo. Le vrai écart est sans doute bien plus grand, mais ce match ne peut pas le mesurer.
- Parties : [docs/match_mcts200_vs_policy.pgn](match_mcts200_vs_policy.pgn). Durée : 47 à 127 demi-coups (médiane 80). Dans 17 parties, le MCTS a eu au moins 3 points de matériel d'avance avant de mater ; 7 parties comportent au moins une promotion.
- **Partie 1 (italienne, MCTS Blancs, 24 coups)** : 12.Qxb7 gagne un pion, puis 22.Ne5 Rc8 23.Nxc6! Rxc6?? 24.Re8#. La politique reprend le cavalier et se fait mater sur la dernière rangée, le motif même qu'elle ne voit pas dans le test du mat du couloir.
- **Partie 12 (est-indienne, MCTS Noirs, 26 coups)** : attaque classique à l'aile roi (…f5-f4, …g5-g4-g3), puis 23…Bxh3! 24.gxh3 Qxh3 25.Rfd1 Qh2+ 26.Kf1 Qh1#, sans avoir eu d'avance de matériel. La politique joue des coups « humains » à l'aile dame (Nb5, a4, Ba5, Nb6) et ne voit pas le mat venir.
- **Partie 5 (française, MCTS Blancs, 127 demi-coups)** : une finale de cavaliers gagnée à la technique (pion de plus, roi actif, promotion puis 64.Qe5#). Elle montre aussi que les gains se convertissent lentement : il n'y a pas de bonus pour un mat plus court.

## Étape 8 (partielle, avant l'étape 6) — Interface UCI et échelle Stockfish (2026-10-20)

### Interface UCI (`src/chesszero/uci.py`, `scripts/uci.py`, `scripts/chesszero.bat`)
- Le MCTS joue dans n'importe quel logiciel UCI. Options : `Simulations` (200), `MoveTime` (ms, 0 = utiliser les simulations), `Threads` (1 à 8, plafond voulu pour ce PC), `BatchSize` (8), `ReuseTree` (oui), `Checkpoint` (défaut `checkpoints/kaggle_sup/best.pt`).
- `go movetime X` ; `go wtime/btime [winc binc movestogo]` : budget = temps restant / 30 (ou / movestogo) + 80 % de l'incrément, jamais plus de la moitié du temps restant, moins 50 ms de marge ; `go nodes N` ; `go infinite` + `stop`. Sans limite : option `MoveTime` si > 0, sinon `Simulations`. Le coup est toujours le plus visité (température 0, sans bruit). **(Comportement remplacé le jour même par les deux modes `UseClock` : voir « Modes de force UCI » plus bas.)**
- La recherche tourne dans un thread, par paquets de 16 simulations : `stop`, `isready`, `quit` répondent tout de suite. Si la recherche plante, un `bestmove` légal est quand même renvoyé (le logiciel ne reste pas bloqué). `info` : profondeur de la variation principale, nodes, nps, temps, `score cp` (courbe de Leela `90·tan(1,5638·Q)`, simple convention d'affichage), pv.
- **Réutilisation de l'arbre** : si la position reçue prolonge celle de l'arbre (notre coup + la réponse adverse), les coups sont joués dans l'arbre (`advance`) et le sous-arbre est gardé ; sinon (`ucinewgame`, autre position, option modifiée, `ReuseTree` désactivé) l'arbre repart de zéro. L'historique complet des coups reste dans l'arbre, donc les répétitions sont détectées.
- stdout ne contient que du protocole ; les messages vont sur stderr.
- Tests : `tests/test_uci.py` (15 tests, évaluateur factice) : poignée de main, nœuds et temps respectés, mat en un (Ra8#), `stop`, réutilisation de l'arbre (même objet, visites conservées), non-réutilisation, historique de répétition, position terminale (`bestmove 0000`), parsing, budget de temps, erreur de recherche.
- Essai réel : `cutechess-cli` lance `chesszero.bat` contre Stockfish avec une pendule 3 s + 0,1 s : les parties se jouent normalement.

### Logiciels installés (hors git, dans `tools/`)
- **Stockfish 19** (release officielle `sf_19`, `stockfish-windows-x86-64-universal.zip`, SHA-256 vérifié contre celui publié par GitHub). `UCI_Elo` va de 1320 à 3190 (constantes `LowestElo` / `HighestElo` de `search.h`), « CCRL Blitz à peu près », calibration à 60 s + 0,6 s par coup d'après la documentation de Stockfish.
- **Cute Chess 1.5.1**, version portable `cutechess-1.5.1-win64.zip` (release officielle GitHub, SHA-256 vérifié). `tools/` est dans le `.gitignore`.
- **Liste de moteurs de Cute Chess** : `scripts/lancer_cutechess.bat` écrit `%APPDATA%\cutechess\engines.json` (« ChessZero - MCTS 200 » et « Stockfish 19 », chemins absolus calculés à partir de l'emplacement du dépôt ; sauvegarde `.bak` d'une ancienne liste) puis lance Cute Chess. **Piège** : l'application Claude Desktop (MSIX) redirige les écritures dans `%APPDATA%` vers `AppData\Local\Packages\Claude_…\LocalCache\Roaming` ; un fichier écrit depuis une session Claude n'est donc pas vu par un Cute Chess lancé normalement. Le `.bat` doit être lancé par double-clic depuis l'explorateur.

### Calcul d'Elo (`src/chesszero/rating.py`, `tests/test_rating.py`, 10 tests)
- Un seul Elo `R` est ajusté sur toutes les parties : score espéré `p = 1 / (1 + 10^((S − R)/400))` contre un niveau `S`, nulle = 0,5, estimation par quasi-vraisemblance de Bernoulli (`Σ(s − p) = 0`, bissection).
- Écart-type = le plus grand de l'écart modèle (`1/(c·√Σp(1−p))`) et de l'écart « sandwich » observé (`√Σ(s−p)² / (c·Σp(1−p))`), `c = ln10/400`. Intervalle à 95 % = ± 1,96 écart-type. Vérifié par simulation : couverture de 90 à 99,5 % pour un nominal de 95 % (300 tirages). Tout gagné ou tout perdu : borne seulement, sans intervalle.

### Protocole de référence (`scripts/match_stockfish.py`, `PROTOCOL` = `stockfish-ladder-v1`)
**À ne pas modifier : les progrès du self-play (étapes 6 et 7) se mesurent en rejouant exactement ceci.**
- **IA** : MCTS, section `[mcts]` de `configs/kaggle.toml` (200 simulations par coup, `c_puct` 1,5, lots de 8, perte virtuelle 1, FPU −0,2), température 0, sans bruit de Dirichlet, arbre réutilisé d'un coup à l'autre, un processus par partie, 2 threads torch. Les deux camps sont joués par `mcts.advance`.
- **Stockfish 19** : `go movetime 300` (300 ms par coup), `Threads 1`, `Hash 16`, `UCI_LimitStrength true`, `UCI_Elo` = le niveau testé ; `ucinewgame` à chaque partie.
- **Ouvertures** : les 10 de `scripts/match.py` (italienne, sicilienne, française, Caro-Kann, gambit dame refusé, est-indienne, anglaise, espagnole, scandinave, London ; 6 demi-coups chacune), chacune avec les deux couleurs : **un tour = 20 parties**.
- **Fin de partie** : mat, pat, matériel insuffisant, 50 coups, triple répétition (règles officielles, historique complet) ; nulle au-delà de 400 demi-coups (jamais atteint dans le premier run). Pas d'abandon ni d'arbitrage.
- **Phase 1** : niveaux 1320, 1600, 1900, 2200, 2500, un tour chacun (100 parties). **Extension** : tant que l'IA marque plus de 80 % au niveau le plus haut joué (phase 1 ou extension), un niveau 300 Elo au-dessus (2800, 3100, plafond 3190), un tour chacun. **Phase 2 (affinage)** : Elo ajusté sur la phase 1 et l'extension, arrondi à 50 → centre ; niveaux centre −100, centre, centre +100 (bornés à 1320–3190), 2 tours de plus chacun (40 parties par niveau, mêmes ouvertures : Stockfish tire au hasard, les parties diffèrent). Elo final = ajustement sur toutes les parties.
- **Machine** : Intel Core i9-13900H (20 threads logiques), Windows 11 Pro, Python 3.12.10, torch 2.14.1+cpu, **4 processus en parallèle × 2 threads torch + Stockfish à 1 thread = 8 threads au plus**. Ne rien lancer d'autre pendant le match : le nombre de simulations ne change pas, mais la force de Stockfish à 300 ms dépend de la vitesse du CPU.
- **Relancer sur un autre checkpoint** (mêmes parties : niveaux, tours, ouvertures, couleurs ; refuse si le protocole a changé, `--force` pour passer outre) :
  `.venv\Scripts\python.exe scripts\match_stockfish.py run --tag <nom> --checkpoint <chemin\best.pt> --like kaggle_sup`
  puis comparaison automatique ; `match_stockfish.py compare kaggle_sup <nom>` pour la refaire. Sans `--like`, le protocole adaptatif est rejoué depuis zéro (les niveaux de la phase 2 dépendent alors du checkpoint ; l'Elo reste comparable car ajusté sur toutes les parties). Si un checkpoint devient beaucoup plus fort, préférer le protocole adaptatif : le rejeu exact serait saturé (le rapport affiche alors `ATTENTION`).
- Résultats : `docs/stockfish/<tag>.json` (protocole, environnement, toutes les parties avec leurs coups), `.pgn`, `.log`. Reprise automatique si le run est interrompu.
- Limite connue : Stockfish n'est pas déterministe, donc deux runs du même protocole ne donnent pas les mêmes parties (ce bruit est dans l'intervalle de confiance).

### Mesure de durée avant lancement (4 parties simultanées, rien enregistré)
0,68 s par coup de l'IA avec 2 threads (0,45 s avec 14 threads au match précédent), partie moyenne 36 s sur ces 4 parties (niveaux 1320, 1900, 2200, 2500), soit 33 min estimées pour 220 parties. **Réel : 52,8 min** : les parties étaient plus longues (moyenne 57 s, 109 demi-coups) que dans l'échantillon, qui comptait une partie de 15 demi-coups. L'extension n'a pas été déclenchée (score de 52,5 % à 2500) ; un niveau d'extension de plus ajouterait 20 parties, environ 5 min.

### Résultat : réseau kaggle (pas 92 000), MCTS-200 — tag `kaggle_sup`
220 parties : 127 gains (toutes par mat), 49 défaites (toutes par mat), 44 nulles (29 triples répétitions, 8 matériels insuffisants, 7 pats).

| niveau Stockfish | parties | V | N | D | score | attendu par le modèle (Elo 2436) | Elo implicite (95 %) |
|---|---|---|---|---|---|---|---|
| 1320 | 20 | 20 | 0 | 0 | 100,0 % | 99,8 % | non mesurable |
| 1600 | 20 | 19 | 1 | 0 | 97,5 % | 99,2 % | 2236 (2039 à 2800) |
| 1900 | 20 | 15 | 2 | 3 | 80,0 % | 95,6 % | 2141 (1996 à 2477) |
| 2200 | 20 | 13 | 3 | 4 | 72,5 % | 79,6 % | 2368 (2231 à 2593) |
| 2300 | 40 | 21 | 12 | 7 | 67,5 % | 68,7 % | 2427 (2339 à 2535) |
| 2400 | 40 | 17 | 10 | 13 | 55,0 % | 55,2 % | 2435 (2340 à 2535) |
| 2500 | 60 | 22 | 16 | 22 | 50,0 % | 41,0 % | 2500 (2423 à 2577) |

- Phase 1 seule : 2389 ± 115 (2275 à 2504) ; la phase 2 a donc porté sur 2300, 2400 et 2500.
- **Elo estimé : 2436 ± 56 (intervalle à 95 % : 2381 à 2492), sur l'échelle UCI_Elo de Stockfish 19** (CCRL Blitz, 300 ms par coup). Sensibilité : 2438 en gardant les niveaux ≥ 1900, 2453 pour ≥ 2200, 2462 ± 60 pour 2300-2500 ; par couleur : 2432 (Blancs) et 2441 (Noirs), donc pas d'effet de couleur.
- Le modèle logistique s'ajuste bien de 2200 à 2500. Contre les niveaux faibles, l'IA marque moins que prévu à 1900 (80 % pour 95,6 % attendu, 3 défaites) : des gaffes tactiques isolées, voir ci-dessous.
- Ouvertures les plus difficiles pour l'IA (niveaux ≥ 2300, score moyen) : gambit dame refusé 29 %, London 43 %, Caro-Kann 46 % ; les meilleures : française 71 %, sicilienne 68 %, scandinave 68 %.
- **Nulles** : analysées avec Stockfish plein (250 ms) 30 demi-coups avant la fin : 37 sur 44 sont dans des positions égales (|éval| < 1,5 pion), 7 avec une avance de l'IA supérieure à 1,5 pion (5 triples répétitions, 1 matériel insuffisant, 1 pat), aucune avec un retard. Les nulles ne sont donc, en majorité, pas des gains gâchés ; la limite « pas de préférence pour le mat le plus court » (étape 5) joue sans doute dans les 7 autres.

### Limites de l'échelle (à relire avant d'interpréter le chiffre)
1. C'est l'échelle **CCRL Blitz de Stockfish**, pas FIDE ni Lichess : le décalage avec ces échelles est incertain (de quelques dizaines à plus d'une centaine de points). Ne pas annoncer « 2436 FIDE ».
2. La calibration de Stockfish suppose 60 s + 0,6 s par coup ; ici 300 ms. À ce temps Stockfish est un peu plus faible que son niveau nominal, surtout aux niveaux hauts : l'Elo de l'IA est donc probablement un peu surestimé par rapport à l'échelle nominale.
3. Stockfish bridé joue des coups parfaits mêlés à des gaffes tirées au hasard, pas comme un humain de ce niveau ; l'IA, tactique, s'en sort sans doute mieux qu'un humain de même force (et gaffe sur des coups simples, voir la partie 40).
4. 10 ouvertures seulement, rejouées 1 à 3 fois : les parties d'une même ouverture ne sont pas indépendantes, l'intervalle est un peu optimiste. 220 parties donnent ± 56 : une différence entre deux checkpoints plus petite que ~80 Elo ne serait pas significative.
5. Dépend de la machine : à 200 simulations l'IA joue la même chose partout, mais la force de Stockfish à 300 ms varie avec la vitesse du CPU.

### Parties marquantes (PGN : `docs/stockfish/kaggle_sup.pgn`, « Round » = numéro de partie dans l'ordre de fin)
- **Partie 201 (niveau 2500, sicilienne, IA Blancs, 49 demi-coups)** : victoire d'attaque. 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 e5 5.Bb5+ Nd7 6.Nf5 a6 7.Bxd7+ Qxd7 8.Nc3 Nf6 9.Bg5 Nxe4 10.Nxg7+ Bxg7 11.Nxe4 d5 12.Nf6+ Bxf6 13.Bxf6 Rg8 14.Qe2 Rxg2 15.Qxe5+ Kf8 16.O-O-O Qe6 17.Qf4 Rg4 18.Qh6+ Ke8 19.Rhe1 Re4 20.Rxe4 Qxe4 21.Bc3 Qf5 22.Bb4 b6 23.Qf8+ Kd7 24.Qd6+ Ke8 25.Qe7#. Stockfish plein donne l'avantage à l'IA à partir du demi-coup 30 (≈ +3 pions) et voit le mat à partir du demi-coup 45.
- **Partie 40 (niveau 1900, italienne, IA Blancs, 38 demi-coups)** : une défaite contre un niveau bien plus faible. L'IA est à +6 pions (Stockfish plein) après 16.Qe4, puis joue **17.Bxe6??** et subit un mat en trois coups : 17…Qxf2+ 18.Rxf2 Rd1+ 19.Qe1 Rxe1#. Une gaffe tactique de profondeur 3 que 200 simulations n'attrapent pas : la recherche ne vérifie pas assez les échecs et sacrifices de dame adverses. C'est le type d'erreur que le self-play devrait corriger.
- **Partie 90 (niveau 2500, Caro-Kann, IA Noirs, 255 demi-coups)** : la plus longue. Position équilibrée jusqu'au demi-coup 100 (éval ≈ 0), puis l'IA propose l'échange des dames (51…Qe7), laisse le pion h promouvoir (53.h7 puis 54.h8=Q) et Stockfish convertit dame contre cavalier en 128.Qa7#. Illustre la faiblesse en finale : le réseau, entraîné sur des coups humains, calcule mal les courses de pions lointaines à 200 simulations.

## Étape 8 (suite) — Modes de force UCI (`UseClock`) et profils Cute Chess (2026-10-20)

### Diagnostic : ce que faisait l'interface avant le changement
Règle de priorité telle qu'elle était codée dans `_cmd_go` (première condition vraie) :
1. `go nodes N` → N simulations (et, si le `go` portait aussi un temps, le temps s'ajoutait comme second plafond) ;
2. `go infinite` ou temps de pendule / `movetime` → **le temps l'emporte, l'option `Simulations` est ignorée** ;
3. `go` nu (aucun temps) et `MoveTime` > 0 → `MoveTime` ms ;
4. sinon → `Simulations`.
Avec Cute Chess, qui envoie toujours `wtime/btime` (ou `movetime`, ou `infinite`), `Simulations` et `MoveTime` n'étaient donc **jamais lus** : l'IA jouait « à la pendule ». Ce n'était pas le comportement de la mesure de référence (200 simulations fixes).

Ce que Cute Chess envoie réellement (journal des commandes reçues, `CHESSZERO_UCI_LOG`) :
- pendule 5 min + 3 s : `go wtime 303000 btime 303000 winc 3000 binc 3000` (l'incrément est ajouté d'avance) ;
- « temps par coup » 5 s : `go movetime 5000` ; 40 coups en 60 s : `go wtime 60000 btime 60000 movestogo 40` ;
- **pendule « infinie » : `go infinite`**, et Cute Chess attend un `bestmove` sans jamais envoyer `stop` : un moteur qui obéit à la lettre ne joue jamais (c'est ce qui bloquait un de mes essais en ligne de commande).

### Vitesse mesurée (moteur UCI réel, `scripts/chesszero.bat`, position initiale et milieu de jeu)
| threads | simulations/s | 200 simulations |
|---|---|---|
| 8 (défaut du moteur) | 354 (début), 348 (milieu) | 0,51 s |
| 2 (comme dans la mesure Stockfish) | 295, 299 | 0,67 s |

Budget de temps à 5 min + 3 s (formule du code : reste/30 + 0,8 × incrément, moins 50 ms, plafonné à la moitié du reste), à 350 simulations/s et en supposant que l'IA dépense tout son budget (les nœuds de l'arbre réutilisé s'ajoutent en plus) : le reste de pendule converge vers ~18 s et le budget vers ~3 s.

| coup | reste (s) | budget (s) | simulations | par rapport aux 200 |
|---|---|---|---|---|
| 1 | 300 | 12,4 | ~4 300 | × 21 |
| 10 | 226 | 9,9 | ~3 500 | × 17 |
| 20 | 166 | 7,9 | ~2 800 | × 14 |
| 40 | 94 | 5,5 | ~1 900 | × 9,6 |
| 80 | 38 | 3,6 | ~1 300 | × 6,3 |

Autres pendules (coup 1 → coup 40) : 1 min + 1 s ≈ 960 → 510 simulations ; 3 min + 2 s ≈ 2 600 → 1 200 ; 10 min + 5 s ≈ 8 400 → 3 500. **En mode pendule, l'IA joue donc avec 6 à 20 fois plus de simulations que dans la mesure de référence : son Elo n'est pas le 2436 mesuré.**

### Changement : option `UseClock` (décochée par défaut) et fonction `search_limits`
- **Force fixe (`UseClock` = false, défaut)** : exactement `Simulations` **nouvelles** simulations par coup (comme `match.py` / `match_stockfish.py` : `mcts.search(N)` ajoute N visites, les visites de l'arbre réutilisé s'y ajoutent), quelle que soit la pendule. `wtime/btime/winc/binc/movetime` sont ignorés. `go nodes N` remplace `Simulations`. **`go infinite` est traité comme une demande de `Simulations` simulations** (écart volontaire au protocole : sinon la pendule « infinie » de Cute Chess ne démarre jamais).
- **Pendule (`UseClock` = true)** : `go movetime X` → X ms ; `go wtime…` → reste/30 (ou /movestogo) + 0,8 × incrément, ≤ la moitié du reste, moins 50 ms ; `Simulations` ignoré ; `go nodes` avec un temps → le premier des deux plafonds ; `go infinite` → jusqu'à `stop`.
- Dans les deux modes : un `go` sans aucune limite utilise `MoveTime` (ms) si > 0, sinon `Simulations` ; `stop` et `quit` interrompent la recherche tout de suite (paquets de 16 simulations ≈ 50 ms). `MoveTime` n'a donc d'effet que sur un `go` nu, que Cute Chess n'envoie jamais.
- Priorité exacte : `infinite` > pendule (si `UseClock`) > `nodes` > `MoveTime` (go nu) > `Simulations`.
- Arguments de ligne de commande de `scripts/uci.py` : `--simulations N`, `--use-clock`, `--move-time`, `--threads`, `--batch-size`, `--checkpoint` : valeurs par défaut d'un profil, annoncées dans les lignes `option … default …` (le logiciel peut ensuite les changer par `setoption`). Variable d'environnement `CHESSZERO_UCI_LOG=<fichier>` : copie horodatée des commandes reçues (diagnostic).
- Tests (`tests/test_uci.py`, 33 tests au total, dont 18 nouveaux cas) : défaut décoché et annoncé ; table de priorité des deux modes (11 cas) ; `MoveTime` seulement pour un `go` nu ; force fixe = exactement 160 simulations avec une horloge de 50 ms (évaluateur ralenti à 10 ms par appel) et 160 de plus au coup suivant, `movetime` ignoré ; mode pendule = arrêt au budget, `Simulations` ignoré ; `go infinite` répond en force fixe ; **`stop` et `quit` répondent en moins d'une seconde pendant une recherche de 1 000 000 simulations** ; arguments de profil et plafond de threads.

### Profils dans Cute Chess (`scripts/lancer_cutechess.bat`)
- Un même moteur, un petit lanceur par profil : `scripts/chesszero_50.bat`, `chesszero_200.bat`, `chesszero_800.bat` (force fixe, `--simulations N`) et `chesszero_pendule.bat` (`--use-clock`) ; la liste de moteurs contient « ChessZero - 50 sim », « ChessZero - 200 sim (référence) », « ChessZero - 800 sim », « ChessZero - pendule » et « Stockfish 19 ».
- **Pourquoi des lanceurs et pas des arguments dans `engines.json`** : essayé en ligne de commande, Cute Chess 1.5.1 **ignore** la clé `arguments` (chaîne ou tableau) et refuse des arguments dans `command` ; `initStrings` est honorée, mais « Configure » afficherait alors 200 et pourrait écraser le profil. Avec un lanceur par profil, les valeurs par défaut annoncées par le moteur sont les bonnes.
- Le `.bat` regénère la liste s'il ne trouve pas « 800 sim » dans le `engines.json` existant (l'ancien est sauvegardé en `engines.json.bak`).
- Vérifié avec `cutechess-cli` (profils entre eux, pendule « infinie ») : 50 sim = 0,21 s par coup, 800 sim = 2,3 s, 200 sim = 0,55 s même avec 20 s + 1 s, pendule à 20 s + 1 s = 1,5 s. Écran de l'interface graphique non vérifié (seule la ligne de commande l'a été).

### Niveaux : ce qui est mesuré et ce qui est estimé
- **Mesuré** : 200 simulations (profil référence, 2 threads, Stockfish 300 ms) = **2436 ± 56 sur l'échelle UCI_Elo de Stockfish 19** (voir plus haut pour les limites de cette échelle) ; MCTS-200 bat la politique seule 20-0 (écart ≥ +317 Elo, borne à 95 %).
- **Estimé, non mesuré** : 50 simulations ≈ 2200 à 2350 (ordre de grandeur : quatre fois moins de simulations ≈ −100 à −250 Elo) ; 800 simulations ≈ 2550 à 2650 (quatre fois plus ≈ +100 à +200, gain qui décroît à mesure que la recherche approche les limites du réseau) ; pendule 5 min + 3 s : plus de 800 simulations presque tout le temps, donc au moins le niveau 800, probablement au-dessus (non estimable sérieusement). Ces fourchettes viennent de l'expérience générale du passage à l'échelle du MCTS, pas de parties jouées par ce réseau. Pour les mesurer : adapter `PROTOCOL["simulations"]` (nouveau protocole = nouvelle référence, une heure environ par profil).
- Aucun de ces profils n'est faible pour un joueur de club : même 50 simulations est annoncé > 2200 sur l'échelle de Stockfish (qui n'est pas l'échelle FIDE).

### Conseils pour jouer (Cute Chess)
- **Pendule** : en force fixe, l'IA ignore l'horloge mais **Cute Chess la décompte quand même** : une IA qui met 2,3 s par coup (800 simulations) perd au temps avec une pendule trop courte (par construction ; je n'ai pas rejoué ce cas jusqu'au bout). Choisir **« Infinite »** (aucune pendule, l'IA répond après N simulations) ou une pendule large : 10 min + 5 s, ou 5 s par coup.
- Ne pas utiliser « Infinite » avec « ChessZero - pendule » : en mode pendule `go infinite` dure jusqu'à `stop`, que Cute Chess n'envoie jamais.
- Ne pas monter « Threads » au-delà de 8 (plafond du moteur) ni lancer d'autres charges lourdes pendant qu'on joue.

## Avant l'étape 6 — Protocole de mesure v2 et tables de finales Syzygy (2026-10-20)

### Protocole v2 (`stockfish-ladder-v2`, `scripts/match_stockfish.py` + `scripts/ladder_openings.py`)
- **Seul changement par rapport à v1 : les ouvertures** (et, en conséquence, le nombre de tours d'affinage). Le reste est identique : 200 simulations, `[mcts]` de `kaggle.toml`, Stockfish 19 à 300 ms / 1 thread / Hash 16 / sans tables, niveaux 1320-2500, extension +300 au-delà de 80 %, affinage centré sur l'Elo de la phase 1 (−100, 0, +100), 4 processus × 2 threads torch, 400 demi-coups au plus. Un test (`tests/test_ladder.py`) vérifie que v1 est exactement le protocole archivé et que v2 n'en diffère que par `name`, `openings` et `refine_rounds`.
- **40 ouvertures, 6 à 10 demi-coups** : 32 lignes courantes (15 en 1.e4, 14 en 1.d4, 3 de flanc) et 8 débuts rares (1.b4, 1.g4, 1.f4, 1.b3, 1.Nc3, 1…Nc6, 1…b6, Budapest). Critère : écart de moins d'un pion selon Stockfish 19 plein (8 threads, 2 s). Écartées pour cette raison : l'Englund (+1,79) et le Saint-Georges (+1,07) ; Tchigorine et Taimanov écartées pour revenir à 40. La London de v1 y est à l'identique.
- **Un tour = 80 parties** (40 × 2 couleurs) : affinage en **1 tour** (80 parties par niveau, contre 40 en v1), chaque ouverture et chaque couleur une fois par niveau. Total sans extension : 640 parties (400 en phase 1 + 240 en affinage).
- **Résultats et IA séparés du protocole** : `docs/stockfish/<v1|v2>/<tag>.json` ; bloc `ai` = options de l'IA hors protocole (`syzygy`, `syzygy_pieces`). Tags de la forme `v1/kaggle_sup` acceptés partout. La mesure v1 est archivée dans `docs/stockfish/v1/` (rapport inchangé : 2436 ± 56).
- Durée estimée avant lancement (16 parties variées, rien enregistré) : 135 à 160 min ; **réel : 142,3 min**.

### Tables Syzygy (`src/chesszero/syzygy.py`, `tools/syzygy/`, hors git)
- **Téléchargement** : 3, 4 et 5 pièces, 290 fichiers, **984,0 Mo** (WDL 396 Mo + DTZ 588 Mo), depuis le miroir Lichess `https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/` et `.../3-4-5-dtz/` ; **les 290 sommes SHA-256 sont conformes** au fichier `sha256` publié à côté. Pour mémoire : 6 pièces = 160 Go, 7 pièces ≈ 18 To.
- **Dans l'arbre** (`MCTS(..., tablebase=)`) : une nouvelle feuille à ≤ 5 pièces sans droit de roque reçoit une valeur exacte (+1 / 0 / −1) sans appel au réseau. Règle des 50 coups : avec un compteur `c` > 0, un gain n'est retenu que si `|DTZ| + c ≤ 100` (comme Stockfish), sinon gain « maudit » = nulle. Cache des sondages par (position, compteur). Si la racine avance dans une feuille exacte qui n'est pas une fin de partie, elle redevient une racine à chercher.
- **À la racine couverte** : `Tablebase.choose` (logique `root_probe` de Stockfish) : mat immédiat, sinon coup gagnant de plus petite DTZ ; en nulle, le coup non perdant le plus visité par le MCTS ; en perte, la défense la plus longue. **Ce n'est pas le mat le plus court** (Syzygy ne donne pas la distance au mat) : c'est la conversion garantie dans la limite des 50 coups.
- Vitesse : sondage WDL ≈ 0,05 à 1,5 ms ; classement des coups d'une racine ≈ 1 à 60 ms ; **≈ 2 s la première fois qu'un fichier est ouvert** (une fois par processus). Sans effet sur la mesure (nombre de simulations fixe).
- UCI : option `SyzygyPath` (défaut `tools/syzygy` s'il existe, `<empty>` sinon) : les profils Cute Chess jouent avec les tables.
- Tests (`tests/test_syzygy.py`, 15) : faux sondeur (couverture, 50 coups, cache, table absente, mat en un, plus petite DTZ, nulle parmi les coups non perdants, défense la plus longue, feuilles exactes et racine dans le MCTS) ; vraies tables (position d'exemple de python-chess : perte, DTZ −53 ; KBN contre K mené au mat en suivant les tables des deux côtés).

### Mesure de référence v2 — tag `v2/kaggle_sup` (réseau kaggle pas 92 000, IA avec Syzygy 5 pièces, Stockfish sans)
Lancée depuis une copie figée du dépôt (`git worktree`, commit `e7e9980`) pour que le travail de l'étape 6a ne la touche pas ; aucune autre charge lourde pendant le match. 640 parties, 142,3 min ; partie moyenne 53 s, 106 demi-coups, 0,80 s par coup de l'IA.

| niveau Stockfish | parties | V | N | D | score | Elo implicite (95 %) |
|---|---|---|---|---|---|---|
| 1320 | 80 | 79 | 1 | 0 | 99,4 % | 2201 (2010 à 2520) |
| 1600 | 80 | 74 | 0 | 6 | 92,5 % | 2036 (1926 à 2306) |
| 1900 | 80 | 63 | 10 | 7 | 85,0 % | 2201 (2121 à 2321) |
| 2200 | 80 | 45 | 8 | 27 | 61,3 % | 2280 (2208 à 2359) |
| 2250 | 80 | 51 | 9 | 20 | 69,4 % | 2392 (2320 à 2479) |
| 2350 | 80 | 48 | 13 | 19 | 68,1 % | 2482 (2412 à 2564) |
| 2450 | 80 | 41 | 14 | 25 | 60,0 % | 2520 (2452 à 2595) |
| 2500 | 80 | 37 | 19 | 24 | 58,1 % | 2557 (2491 à 2628) |

- **Elo estimé : 2427 ± 37 (intervalle à 95 % : 2390 à 2464)**, échelle UCI_Elo de Stockfish 19. Phase 1 seule : 2367 ± 63, d'où l'affinage à 2250 / 2350 / 2450. Extension non déclenchée (58,1 % à 2500).
- **Comparaison avec v1** (2436 ± 56, sans tables, 10 ouvertures) : −10 Elo (± 67), pas significatif. Les deux changements (ouvertures, tables) ne sont pas séparables ; d'après l'analyse des finales ci-dessous, les tables ne pèsent presque rien.
- Fins : 438 gains et 128 défaites (tous par mat), 74 nulles (44 triples répétitions, 17 pats, 13 matériels insuffisants).
- Par couleur : Blancs 2468 ± 54, Noirs 2388 ± 51 (écart d'environ 2 écarts-types : les ouvertures retenues donnent en moyenne +0,3 aux Blancs). Courantes (32) 2422 ± 41, rares (8) 2446 ± 81 : **pas de faiblesse visible sur les débuts rares**. Par ouverture (16 parties chacune, très bruité) : plus dures est-indienne classique 59 %, Benoni, française avance, gambit dame accepté, Larsen 62 % ; meilleures viennoise 94 %, Réti 91 %, Nimzowitsch, Pirc autrichienne, polonaise 88 %.
- **Le modèle Elo s'ajuste mal à la forme des résultats** : l'Elo implicite monte avec le niveau (2036 à 1600, 2557 à 2500). Un ajustement avec une pente libre (`p = 1/(1+10^((S−R)/(400·k)))`) est bien meilleur (log-vraisemblance +25,9 pour un paramètre de plus ; rapport de vraisemblance 51,8 ≫ 3,84) avec **k ≈ 2,1** et un point à 50 % vers **2555**, extrapolé au-delà du plus haut niveau joué. Lecture : l'IA perd un peu contre tout le monde (gaffes, voir ci-dessous) et Stockfish bridé fait des erreurs aléatoires que l'IA punit même aux niveaux hauts. **Conséquence : le chiffre unique (2427) dépend des niveaux joués ; il reste comparable entre checkpoints mesurés sur la même échelle (`--like`), mais ce n'est pas une force absolue.** Le même motif était visible en v1 (80 % à 1900 pour 95,6 % attendus).
- **Défaites contre 1600 et 1900 (13 parties)** : longues (67 à 196 demi-coups) ; dans 11 sur 13, l'IA avait été à +1 pion ou mieux (Stockfish plein, 0,1 s). Le coup décisif (première chute ≥ 2 pions qui fait passer sous −1,5) tombe **en milieu de jeu** (15 à 27 pièces) dans 12 sur 13, entre les demi-coups 24 et 75 : gaffes tactiques, pas des problèmes de finale. C'est la cible principale du self-play.

### Finales : `match_stockfish.py endgames` (résultat exact à l'entrée dans les tables → résultat réel)

| mesure | parties à ≤ 5 pièces | gain exact → G / N / P | nulle exacte → G / N / P | perte exacte → G / N / P | points perdus / pris |
|---|---|---|---|---|---|
| v1 (sans tables) | 41 / 220 | 12 → 11 / 1 / 0 | 15 → 1 / 14 / 0 | 14 → 0 / 0 / 14 | 0,5 / 0,5 |
| v2 (avec tables) | 109 / 640 | 52 → 52 / 0 / 0 | 33 → 2 / 31 / 0 | 24 → 0 / 0 / 24 | 0 / 1,0 |

- **En v1, les tables n'auraient sauvé qu'environ 0,5 point sur 220 parties (≈ 1 Elo).** Avec les tables, la conversion des finales gagnées est parfaite (52 sur 52) ; elles ont choisi 1 404 coups de l'IA dans 103 parties. Leur intérêt réel est ailleurs : arbitrage exact et rapide des fins de partie en self-play, cibles de valeur exactes, et fin des gains « jamais convertis » (limite de l'étape 5).
- Limite de l'analyse : elle ne voit pas l'effet des tables comme feuilles exactes dans l'arbre pour des finales à 6-7 pièces ; d'après l'Elo (−10 ± 67), cet effet est petit.

## Étape 6a — Vitesse du MCTS pour le self-play (2026-10-21)

### Profil avant (faux réseau, milieu de jeu, 2 400 simulations, lots de 8)
Coups légaux + indices des coups 40 % (dont `move_to_index` 13 %), encodage `pack_board` 14 %, sélection PUCT (numpy sur de petits tableaux) 14 %, softmax par position dans l'évaluateur ≈ 10 %, `push`/`pop` ≈ 12 %.

### Changements (résultats identiques, vérifiés par tests)
- `encoding.pack_board` : lit les 12 masques de bits de python-chess et les décompresse d'un coup (`np.unpackbits`) au lieu de parcourir `piece_map()`. Test : identique à l'ancienne version sur plus de 3 000 positions aléatoires et des positions spéciales (en passant, roques, promotions).
- `encoding.move_indices` : table précalculée (départ, arrivée) → indice ; sous-promotions par `move_to_index`. Test : identique à `move_to_index` coup par coup.
- `Node.select` : même calcul en moins d'opérations numpy (raccourci exact quand aucune visite : argmax des priors).
- **Recherche en deux temps** : `MCTS.collect(k)` (descentes ; feuilles terminales ou exactes remontées tout de suite ; feuilles à évaluer renvoyées avec leurs coups légaux) puis `MCTS.apply(feuilles, priors, valeurs)`. **`search_batched(arbres, évaluateur, N, feuilles_par_arbre)`** ajoute N simulations à chaque arbre en envoyant les feuilles de tous les arbres dans un même appel au réseau. Test : visites et valeurs **identiques** à des `search` séparés (avec bruit de Dirichlet), lots réellement regroupés.
- `NetEvaluator.evaluate` : envoie les positions compactes (67 octets) au device, les décode là (`encoding.unpack_batch_torch`, testé identique à `unpack_batch`), rassemble les logits des coups légaux et fait le softmax sur le device (fp16 automatique sur CUDA) ; seules les probabilités utiles reviennent. Test : mêmes priors et valeurs que l'ancien calcul (écart < 1e-5).
- `scripts/bench_selfplay.py` : débit du self-play (P processus × G parties, `search_batched`, bruit et température 1 ; `--stub` = faux réseau pour le coût CPU seul), avec estimation des parties par heure pour 200 simulations par coup et deux réglages de « playout cap randomization ». À lancer sur Kaggle en 6e.

### Mesures (i9-13900H, un thread torch par processus)

| | ancien code | nouveau code |
|---|---|---|
| un arbre, lots de 8, faux réseau : position initiale / milieu / finale (simulations/s, 1 cœur) | 5 749 / 3 299 / 5 779 | 6 695 / 3 660 / 6 348 (+10 à +16 %) |
| 64 parties en parallèle, 4 feuilles par arbre (lots de 254), faux réseau, 1 processus | — | 3 569 |
| idem, 3 processus | — | **10 490** (3 497 par processus : montée linéaire) |

- Le gain CPU pur est modeste (le softmax, désormais sur le GPU, n'est pas dans le faux réseau). Après optimisation, le profil est dominé par la sélection (≈ 29 %), les coups légaux de python-chess (≈ 27 %) et `push` (≈ 21 %) : il ne reste plus de gros gain facile en Python.
- Essai avec le vrai réseau sur CPU (16 parties, lots de 61) : la chaîne complète fonctionne ; sur CPU le réseau prend 97 % du temps (attendu : sur GPU, c'est l'inverse).
- **Estimation Kaggle (à mesurer en 6e)** : cœur Xeon environ 2 à 2,5 fois plus lent → 1 400 à 1 750 simulations/s par processus, 2 500 à 4 500 au total avec 3 processus et le coût réel des appels GPU, soit **450 à 800 parties/h à 200 simulations par coup** (100 demi-coups).
- Non fait (gain incertain, à décider après la mesure Kaggle) : recouvrement CPU / GPU (deux moitiés de parties en alternance).

## Étape 6 — Décisions, match de vérification de la PCR, 6b livre d'ouvertures (2026-10-21)

### Décisions de l'auteur
- Taux d'apprentissage du self-play : AdamW neuf, montée linéaire de 2e-5 à 2e-4 sur 300 pas, puis 2e-4 constant ; si la perte de politique de validation Lichess monte de plus de 0,05 après la première génération, diviser par 2.
- « Playout cap randomization » (p = 0,25, 320 simulations complètes enregistrées / 64 rapides non enregistrées, tampon ~250 000 positions, retour à p = 0,5 si le top-1 Lichess baisse de plus de 3 points) : adoptée **sous réserve** du match ci-dessous ; **décision en attente** depuis le résultat.
- Fins de ligne : `.gitattributes` = `* text=auto eol=lf`, `*.bat`/`*.cmd`/`*.ps1` en CRLF, binaires marqués. Avant : `text=auto` seul, donc `core.autocrlf=true` de Git pour Windows convertissait en CRLF à l'extraction (21 fichiers en CRLF, 31 en LF, 3 mélangés). Copie de travail réextraite : tout en LF sauf les 6 `.bat`.

### Match de vérification : MCTS-64 contre MCTS-200 (`scripts/match_sims.py`)
Même réseau (kaggle, pas 92 000), 40 ouvertures v2 × 2 couleurs = 80 parties, arbres réutilisés, température 0, sans bruit, tables Syzygy pour les deux camps, 4 processus × 2 threads, 17 min.
- **MCTS-64 : 6,5 / 80 (8,1 %) : 1 gain, 11 nulles (triples répétitions), 68 défaites (toutes par mat) ; écart −421 ± 139 Elo** (95 %). Par couleur : 64 avec les Blancs 1 G / 6 N / 33 P, avec les Noirs 0 / 5 / 35.
- Le script affichait d'abord « 0 ± 76 » : `fit_elo` cherche dans [0, 4500] et l'adversaire était placé à 0, d'où un écart négatif tronqué à 0. Corrigé (adversaire placé à 2000) ; note ajoutée au `.log`.
- Lecture : la recherche apporte environ 250 Elo par doublement de simulations entre 64 et 200. Avec p = 0,25, trois coups sur quatre seraient joués ~420 Elo plus faibles : les résultats des parties (cibles de valeur) viendraient d'un joueur qui gaffe beaucoup plus, pour seulement +56 % de parties par heure. Le repli p = 0,5 (80 simulations rapides, estimé ~−330, non mesuré) a le même défaut ; une PCR « douce » (280 / 160) ne gagnerait que ~5 % de parties. **Recommandation de Claude : ne pas adopter la PCR** ; 200 simulations à chaque coup, toutes enregistrées, tampon ~500 000 positions ; pour la tête valeur, abandons + arbitrage par les tables, et cible de valeur mélangée (résultat + Q racine) / 2 à évaluer en 6d.

### 6b — Livre d'ouvertures (`src/chesszero/book.py`, `scripts/prepare_book.py`, `books/lichess_2000.*`)
- **Lecture rapide des coups** (`first_moves`) : sans `chess.pgn`, les commentaires `{ [%clk] [%eval] }`, numéros de coups, annotations `?!` et NAG `$n` sont retirés, puis les coups SAN sont joués jusqu'au demi-coup voulu.
- **Coupe** : demi-coup tiré uniformément dans [8, 20], graine = identifiant de la partie (`crc32(Site)`) : même livre à chaque extraction. Il faut au moins un coup joué après la coupe et une position non terminale. On garde **la suite de coups** (historique identique à une vraie partie).
- **Filtres** : ceux des données supervisées (`configs/data.toml` : Elo ≥ 2000 des deux joueurs, pas de bullet, pas de BOT, fin normale ou au temps, partie standard), parties d'entraînement seulement (pas de validation). Une entrée par position distincte (clé EPD), première suite rencontrée, nombre de parties gardé. Puis valeur du réseau (kaggle, pas 92 000) : rejet si |valeur| > 0,3.
- **Extraction réelle** : janvier 2026 (mois non utilisé par le pré-entraînement), lu en flux : 4 859 418 parties lues, 340 250 gardées, 337 772 coupes valides, 261 047 positions distinctes (629 s de lecture, 7 processus) ; valeurs à 539 pos/s ; **livre final 239 361 positions** (21 686 écartées par la valeur), 18,6 min au total. Fichier `books/lichess_2000.tsv.gz` = **4,3 Mo, versionné dans git** (Kaggle le reçoit avec le clone) + `lichess_2000.json` (paramètres, statistiques) + `.log`.
- **Contenu** : demi-coups 8 à 20 de 9 866 (8) à 21 663 (17) positions (les coupes courtes tombent plus souvent sur une position déjà vue) ; premiers coups e4 50 %, d4 31 %, Nf3 6,5 %, c4 5,7 %, b3 1,6 %, f4 1,2 %, g3 1,1 %, puis e3, Nc3, d3, b4, g4… ; valeurs du réseau : médiane 0,006, 5 % à 95 % entre −0,27 et +0,29 ; 90,1 % des positions viennent d'une seule partie (maximum 465) ; 4,1 % des suites passent par une des 40 ouvertures du protocole v2 (pas d'exclusion : seule l'arène du self-play exclura ces 40 positions).
- **Utilisation** : `OpeningBook.load("books/lichess_2000.tsv.gz")` (0,3 s), `book.sample(rng)` = position tirée **uniformément parmi les positions distinctes** (pas selon leur fréquence), avec toute la pile de coups (0,2 ms par tirage). Blancs au trait dans 53 % des tirages.
- Tests : `tests/test_book.py` (6) : lecture d'une partie au format Lichess réel (pendule, éval, roques, annotations, NAG), coup illisible, coupe déterministe et uniforme, suite de coups conservée, aller-retour du fichier et tirage uniforme.

### Suite (2026-10-21) : décision PCR et retrait des ouvertures de la mesure
- **PCR : non adoptée** (décision de l'auteur, sur la recommandation ci-dessus) : 200 simulations à chaque coup, toutes enregistrées ; tampon d'environ 500 000 positions ; cible de valeur mélangée (résultat + Q racine) / 2 à évaluer en 6d.
- **Livre sans les ouvertures du protocole v2** (demande de l'auteur : le self-play ne doit pas s'entraîner sur les ouvertures de la mesure). `book.opening_keys` / `touches_openings` / `OpeningBook.excluding` : on compare des **positions** (`Board._transposition_key()` : pièces, trait, roques, en passant), pas des préfixes de coups. Une entrée est retirée si une position de sa suite est l'une des 40 positions finales v2 (**13 404 entrées**, dont 9 789 par préfixe de coups et 3 615 par transposition, d'où plus que les 4,1 % annoncés avant) ou si sa position finale est sur une ligne v2 avant sa fin (**12**). Appliqué au livre existant par `prepare_book.py --refilter` (21 s ; même résultat qu'une réextraction, les deux filtres étant indépendants), et intégré à la chaîne normale. **Livre final : 225 945 positions** (premiers coups e4 111 938, d4 69 540, Nf3 15 270, c4 12 859, b3 3 571, f4 2 839…). `books/lichess_2000.json` garde le détail (`after_value_filter` 239 361, `excluded_through`, `excluded_on_path`).
- Tests (`tests/test_book.py`, 8) : exclusion par même ordre de coups, par transposition et sur une ligne interrompue, lignes voisines gardées ; **le livre versionné ne touche aucune des 40 ouvertures** (relit les 225 945 suites, ~20 s). Contre-épreuve à la main : la Najdorf v2 complète, la même par transposition et la ligne interrompue sont détectées, la Kan (hors mesure) ne l'est pas.
- Limite : 10 % des parties de self-play partent de la position initiale et peuvent, au fil des coups, rejoindre une position v2 ; c'est inévitable et sans grande portée (la mesure joue contre Stockfish, pas contre des coups appris par cœur). L'arène du self-play, elle, n'utilisera pas ces 40 positions.

## Étape 6c — Moteur de self-play et enregistrement (2026-10-21)

### Choix validés par l'auteur (avant l'implémentation)
- **Format** : un shard `.npz` par lot de 100 parties ou de 10 minutes et par processus, + `.json` (session, processus, graine, génération, checkpoint et son pas, livre, tables, device, réglages `[mcts]` et `[selfplay]`, horodatage, nombre de positions et de parties). Écriture atomique : `.npz` temporaire renommé, puis `.json` renommé ; **un `.json` présent garantit un `.npz` complet** (un shard sans `.json` est ignoré).
  - Positions : `boards` (N, 67) uint8 ; `policy_offsets` (N+1) int32, `policy_index` int16 et `policy_visits` uint16 = **tous les coups légaux de la racine et leurs visites** (cible de politique ET masque des coups légaux : plus de masques à recalculer) ; `played` int16 (coup joué) ; `result` int8 (camp au trait) ; `root_q` float16 (camp au trait, pour la cible de valeur mélangée de 6d) ; `ply` int16 ; `game` int32.
  - Parties : `game_uid`, `game_moves` (coups UCI depuis la position initiale, ligne du livre comprise), `start_index` (n° dans le livre, −1 = position initiale), `start_plies`, `result_white`, `termination` (mat, pat, matériel insuffisant, 50 coups, triple répétition, tables, abandon, limite), `plies`, `control`, `would_resign` (demi-coup où les Blancs / les Noirs auraient abandonné, −1 sinon).
  - Positions à un seul coup légal : coup joué sans recherche, pas d'enregistrement.
- **Abandon** : Q racine ET Q du coup le plus visité sous −0,90 pendant 3 coups d'affilée du même camp, à partir du demi-coup 20 ; la position de l'abandon est enregistrée (cible de politique valable), le coup n'est pas joué. **Parties de contrôle** (10 %) : aucun abandon ; le premier demi-coup où chaque camp aurait abandonné est noté, pour mesurer les faux abandons (le camp n'a finalement pas perdu) ; ajustement du seuil (−0,02 si plus de 5 % de faux abandons sur au moins 200 parties de contrôle) prévu en étape 7.
- **Arbitrage par les tables en mode WDL seul** : la partie s'arrête à la première position à ≤ 5 pièces (sans droit de roque), avec le résultat exact ; **gain « maudit » (WDL +1) et perte « bénie » (−1) = nulle** (`WDL_VALUE`). Le nombre de pièces ne baisse que par une prise, donc cette position a toujours un compteur des 50 coups nul : la WDL y est exacte sans DTZ. En mode WDL seul, une position à compteur non nul est déclarée non couverte (le réseau l'évalue) et le choix du coup par DTZ est désactivé. Sur Kaggle : tables WDL seulement (396 Mo). Tests : faux sondeur (+1, −1, 0 → nulle ; compteur non nul → non couvert ; pas de choix de coup) et **vraies tables** : `8/p7/3N4/8/8/N1K5/6k1/8 w` (deux cavaliers contre pion, WDL +1, DTZ 102) et `8/k7/4p3/2N5/3K4/8/4N3/8 b` (WDL −1) → nulle ; KQ contre K → gain des Blancs quel que soit le camp au trait.
- **Fin de session Kaggle** : budget `--max-minutes` ; plus de nouvelles parties 20 min avant la fin, arrêt 5 min avant (parties inachevées perdues : sans résultat, elles sont inutilisables) ; une coupure brutale ne perd que ce qui n'est pas encore écrit (≤ 100 parties ou 10 min par processus).

### Tampon entre sessions Kaggle (`selfplay.sync_buffer`, demande de l'auteur)
La sortie d'une version de notebook ne contient que `/kaggle/working` de CETTE session ; un tampon qui ne vivrait que dans la sortie de la session où il a été produit serait perdu à la session suivante. Mécanisme :
1. **Début de session** : la sortie de la session précédente est ajoutée comme donnée du notebook (lecture seule, `/kaggle/input/<...>/selfplay`). `scripts/selfplay.py --inputs <ce dossier> --out /kaggle/working/selfplay` appelle `sync_buffer` : on liste les shards complets des entrées et du dossier de travail, on les trie du plus récent au plus ancien (horodatage du `.json`), on garde les plus récents jusqu'à 500 000 positions (le dernier shard retenu peut dépasser un peu), et on **copie** ceux qui manquent dans `/kaggle/working/selfplay`.
2. **Pendant la session** : les nouveaux shards s'écrivent dans `/kaggle/working/selfplay`.
3. **Fin de session** : nouveau `sync_buffer` sur le seul dossier de travail : les nouveaux shards (les plus récents) entrent dans la fenêtre, les plus anciens en sortent et sont **supprimés** du dossier de travail. La sortie de la session contient donc exactement la fenêtre du tampon (environ 110 Mo), plus le checkpoint (étape 7), et devient l'entrée de la session suivante.
- Noms uniques (`<session>_w<processus>_<n°>`), donc pas de collision ; un shard présent à la fois dans l'entrée et dans le dossier de travail n'est compté qu'une fois ; l'entrée n'est jamais modifiée. Tests : fenêtre limitée aux plus récents, reprise puis élagage sur deux « sessions », shard sans `.json` ignoré.

### Code
- `src/chesszero/selfplay.py` : `SelfPlayConfig` (section `[selfplay]` de `kaggle.toml`), `Game` (coups forcés, enregistrement, température filtrée, abandon et contrôle, fin), `end_of_game`, `run_selfplay` (`games_per_worker` parties ensemble, `search_batched`, échéances), `ShardWriter`, `load_shard`, `list_shards` / `buffer_window` / `sync_buffer`, `verify_shard` (chaque partie rejouée depuis la position initiale : positions, coups légaux de la position décodée, coup joué présent, visité et égal au coup de la partie, visites ≥ simulations, signe du résultat, fin cohérente avec la position finale, conditions de l'abandon, demi-coups d'abandon des parties de contrôle).
- `src/chesszero/syzygy.py` : mode `wdl_only`.
- `scripts/selfplay.py` (lanceur multi-processus, tampon au début et à la fin), `scripts/check_selfplay.py` (vérification + résumé + PGN).
- Tests : `tests/test_selfplay.py` (16 ; 194 au total) : parties enregistrées et vérifiées ; corruption détectée (résultat inversé, coup joué, position) ; départs du livre ; abandon (les Blancs perdus abandonnent dès le demi-coup 20) ; parties de contrôle (aucun abandon, demi-coup 20 noté) ; arbitrage par les tables avec un faux sondeur (avec « le camp au trait gagne », le MCTS évitait toute prise ou poussée de pion et la partie finissait par répétition : preuve que les valeurs exactes servent dans l'arbre ; test fait avec « le camp au trait perd ») ; gain maudit et perte bénie ; tampon ; écriture tous les N parties ; échéances.

### Essai local avec le vrai réseau (avant Kaggle)
`scripts/selfplay.py --out data/selfplay_essai --workers 2 --threads 2 --games-per-worker 4 --max-games 4` : réseau kaggle (pas 92 000) sur CPU, 4 threads au total, 200 simulations par coup, livre, tables en WDL seul. **8 parties, 748 positions enregistrées, 3,8 min** (648 simulations/s, 12 positions par appel au réseau, réseau 92 % du temps : sur CPU c'est lui qui limite ; 125 parties/h, chiffre sans rapport avec Kaggle). 2 shards (22 Ko chacun, ~29 octets par position une fois compressés).
- **`check_selfplay.py` : aucune erreur** (positions = parties rejouées, coups légaux de la position décodée = liste enregistrée, coup joué présent, visité et égal à celui de la partie, visites ≥ 200, signe du résultat des deux camps, fins cohérentes, conditions d'abandon, début de chaque partie = sa ligne du livre).
- Longueur : 70 à 143 demi-coups (moyenne 109, dont 93 joués en self-play après le livre, départs entre les demi-coups 11 et 20).
- Fins : **abandon 2** (un gain de chaque camp, Q racine final −0,94 et −0,97), **tables 3** (tour + pion contre tour : nulle ; fou contre deux pions : nulle ; trois pions contre roi : gain des Blancs), **triple répétition 3** (dont une à 70 demi-coups avec presque tout le matériel : les répétitions précoces sont à surveiller dans le taux de nulles du self-play). Résultats : Blancs 2, nulles 5, Noirs 1.
- Diversité : 8 départs du livre, 8 positions de départ distinctes (ouverture du roi, Petrov, Saint-Georges, espagnole d'échange, London, sicilienne Rossolimo, Caro-Kann London, slave), 8 positions distinctes au demi-coup 30.
- Parties de contrôle : 2 ; un abandon se serait déclenché une fois (Noirs, demi-coup 101) et les Noirs ont bien perdu : 0 faux abandon.
- Visites à la racine : 200 à 1 405, moyenne 499 : la réutilisation de l'arbre ajoute en moyenne ~300 visites aux 200 nouvelles simulations (la cible de politique profite de tout l'arbre).

### Indicateur des répétitions précoces (2026-10-21, demande de l'auteur)
- `selfplay.repetition_stats` / `describe_repetitions` : parties nulles par triple répétition (part des parties et des nulles), réparties par demi-coup de fin (< 60, 60-99, 100-149, 150-199, ≥ 200) et par matériel restant (points P 1, N/B 3, R 5, Q 9 ; 78 dans la position initiale : ≥ 60 presque complet, 40-59, 20-39, < 20 finale), médianes. Calculé à la fin de chaque session sur les parties de la session : affiché et écrit dans `<out>/sessions/<session>.json` (avec débit, fins, parties de contrôle et faux abandons ; ce fichier n'a pas de `.npz` et n'est donc jamais pris pour un shard). Aussi dans `check_selfplay.py`.
- Sur l'essai local (8 parties) : 3 répétitions = 37,5 % des parties, 60 % des nulles ; fins aux demi-coups 70, 127 et 132 ; matériel 40-59 pour une (milieu de jeu), < 20 pour deux (finales). Trop peu de parties pour conclure ; **décision sur la double répétition dans l'arbre après les premières sessions Kaggle**.
- Test : `test_repetition_stats_by_ply_and_material` (répétition par va-et-vient des cavaliers au demi-coup 8, matériel 78).

## Étape 6d — Entraînement sur le self-play : tampon, mélange humain, garde-fous (2026-10-22)

### Choix validés par l'auteur
- **Tirage** : uniforme sur toutes les positions d'entraînement de la fenêtre du tampon (~500 000), sans favoriser les récentes (la fenêtre glissante suffit ; un tirage biaisé vers les dernières parties concentrerait l'apprentissage sur peu de parties corrélées).
- **Pas par tour** : chaque nouvelle position (pas encore utilisée par un tour accepté) vue `reuse` = 4 fois en moyenne : pas = ⌈4 × nouvelles / positions de self-play par lot⌉ (768 à 25 % humain, soit 1 pas pour 192 nouvelles positions ; ~520 pas pour 100 000).
- **Premier tour** (demande de l'auteur : éviter 4 passages en un seul tour sur un petit tampon) : aucun tour tant que le tampon a moins de **250 000 positions d'entraînement** (~3,5 à 6 h de self-play du réseau de départ), ET plafond de **1 passage moyen par position du tampon dans un même tour** (pas ≤ tampon / 768) : au premier tour, 325 pas au plus au lieu de 1 300 ; ensuite (100 000 nouvelles pour 500 000) le plafond ne joue plus. Les positions des premiers tours restent plusieurs tours dans la fenêtre et finissent autour de 4 passages, étalés sur plusieurs réseaux.
- **Mélange Lichess** : composition EXACTE de chaque lot (pas un tirage au hasard) : `round(1024 × part)` positions de `lichess_24m` (partie entraînement, tirage uniforme ; politique = coup joué, valeur = résultat si `value_valid`), le reste du tampon ; **25 %, puis 10 % à partir de la 6e génération acceptée** ; perte moyenne par position (chaque part pèse selon sa place dans le lot).
- **Taux** (décidé avant) : AdamW neuf, 2e-5 → 2e-4 sur 300 pas, puis constant, × `lr_scale`.
- **Cible de valeur mélangée** : option `value_target = "blend"` = (résultat + Q racine) / 2 ; évaluation par `scripts/eval_value_target.py` (ci-dessous) ; **décision seulement si les deux mesures vont dans le même sens** (demande de l'auteur).
- **Garde-fous révisés par l'auteur** (le self-play doit s'éloigner de l'imitation humaine : des seuils fixes par rapport au réseau supervisé finiraient par rejeter tous les candidats parce qu'ils progressent) :
  - pendant les 5 premières générations : G1 (perte de politique Lichess > supervisé + 0,05 → rejet, taux ÷ 2, plancher 2,5e-5) et G2 (top-1 Lichess < supervisé − 3 points → rejet, part humaine remise et maintenue à 25 %) ;
  - ensuite : rejet seulement en cas de chute catastrophique (top-1 −10 points ou perte de politique +0,15 → rejet, taux ÷ 2) ; les dépassements de G1 / G2 deviennent des alertes ;
  - chute brutale par rapport à la génération précédente (top-1 −2 points ou perte de politique +0,05) : alerte ;
  - G3 : perte de valeur Lichess > supervisé + 0,05 : alerte ; **surapprentissage strict** : erreur de valeur des parties de self-play mises de côté en hausse deux tours de suite pendant que celle d'entraînement baisse → rejet, réutilisation ramenée à 2 ;
  - G4 : 3 rejets d'affilée → arrêt (`status = stop` dans `train_state.pt`, le lancement suivant refuse de continuer), décision humaine.
  - Un rejet ramène modèle ET optimiseur au dernier réseau accepté (`accepted.pt`) ; les nouvelles positions ne sont pas marquées comme utilisées : le tour suivant les reprend.
- **Remarques de Claude sur le raisonnement de l'auteur** (validé, avec deux compléments pour l'étape 7) : une arène « nouveau contre précédent » seule mesure un progrès relatif ; de petites régressions non significatives (± 50 Elo à 200 parties) peuvent s'accumuler et le progrès peut être non transitif. Proposé pour l'étape 7 : arène aussi contre le réseau supervisé (ancre fixe) et protocole v2 contre Stockfish toutes les ~5 générations. Limite du critère de surapprentissage : les parties mises de côté changent quand la fenêtre glisse (comparaisons entre ensembles différents) ; exiger deux hausses consécutives limite les fausses alertes.

### Code
- `src/chesszero/selfplay_train.py` : `SelfPlayTrainConfig` (section `[selfplay_train]` de `kaggle.toml`), `is_heldout` (10 % des parties, par `crc32(game_uid)` : stable), `SelfPlayData` (fenêtre en mémoire, cibles denses à partir des visites), `human_val_set` (sous-ensemble fixe de 50 000 positions Lichess avec masques), `batch_rows` (lot n° `step` du tour `round_id` : ne dépend que de (graine, tour, pas)), `make_batch`, `mixed_loss` (perte + détail self-play / humain), `eval_human`, `eval_selfplay` (erreur de valeur contre le RÉSULTAT des parties, quelle que soit la cible), `decide` (fonction pure), `plan_steps`, `lr_at`, `SelfPlayTrainer` (`train_state.pt` tous les 100 pas, `accepted.pt`, `generations/gen_XXXX.pt` au format habituel, `rounds.jsonl`, `train_log.jsonl`). Masques Lichess calculés par un pool, 6 lots à l'avance.
- `scripts/train_selfplay.py` : un tour par lancement (relancer = reprendre) ; codes de sortie 0 accepté / en attente, 1 rejeté, 2 arrêt G4, 3 interrompu.
- `scripts/eval_value_target.py` : `label` (jeu fixe : positions des parties de self-play mises de côté + validation Lichess à `value_valid`, chacune notée par Stockfish plein, score attendu = 2 × espérance WDL − 1, camp au trait) et `compare A B` (erreur contre Stockfish, toutes positions et par source, et contre le résultat réel des parties mises de côté ; bootstrap apparié par partie, 2 000 tirages ; verdict « adopter » seulement si les deux mesures donnent le mélange meilleur et que l'écart contre Stockfish est significatif ; si elles se contredisent : garder le résultat seul).
- Tests (`tests/test_selfplay_train.py`, 13) : pas par tour (tampon minimal, plafond du premier tour, régime normal), composition exacte et déterminisme des lots, garde-fous (G1 et G2 stricts au début ; après 5 générations alertes seulement et rejet sur chute catastrophique ; chute brutale et G3 en alertes ; surapprentissage → rejet et réutilisation réduite ; pas de rejet si l'erreur des parties mises de côté ne monte pas deux fois), tour complet (part humaine exacte à chaque lot, perte en baisse), **reprise après coupure = mêmes poids (`torch.equal`)**, génération acceptée rechargeable par `load_for_inference`, rejet = retour aux poids acceptés + taux ÷ 2, G4 = arrêt après 3 rejets, attente si tampon trop petit, bootstrap par partie.

### Test local (demande de l'auteur)
`train_selfplay.py --buffer data/selfplay_essai --human data/lichess_2000 --batch-size 64 --steps 300 --force --lichess-val 2000 --threads 4 --workers 2` : réseau kaggle (pas 92 000) sur CPU, les 748 positions de l'essai de self-play (8 parties) + Lichess (2,94 M positions d'entraînement chargées), lots de 64 = 48 de self-play + 16 Lichess, 300 pas, 1 min 34 (chargement compris).
- **La perte baisse** : self-play politique 1,759 → 1,389, valeur 0,152 → 0,027 ; perte totale 1,914 → 1,538 (pas 20 → 300). Sur 748 positions vues ~19 fois chacune, c'est surtout de la mémorisation, attendue ici ; la politique humaine reste stable (1,364 → 1,391).
- **Part humaine exacte** : 0,250 à chaque ligne du journal (et contrôlée par une assertion à chaque lot : 16 sur 64).
- **Reprise** : même tour coupé au pas 150 (`--stop-after 150`, code de sortie 3) puis relancé : **poids et état de l'optimiseur identiques au bit près** (`torch.equal`) à ceux du tour d'une traite, mêmes métriques finales. Premier essai : une seule ligne du journal différait (pas 160 : sa fenêtre de moyenne ne couvrait plus que les pas 151-160) ; la fenêtre en cours est maintenant enregistrée avec le tour, et le journal est lui aussi identique (vérifié sur le vrai réseau et dans le test).
- **Décision** : accepté (génération 1) : validation Lichess (2 000 positions) politique 1,4046 → 1,4432 (+0,039, sous le seuil G1 de 0,05), top-1 52,9 % → 51,4 % (−1,5 point, sous G2), valeur 0,684 → 0,702 (+0,018, sous l'alerte G3). Aucune partie mise de côté parmi les 8 (10 % par identifiant) : le critère de surapprentissage n'a pas pu jouer ici.
- `eval_value_target.py` sur un mini-jeu (100 positions de l'essai + 100 Lichess, Stockfish 50 ms, 4 s) : la chaîne marche (estimation WDL de Stockfish — il faut l'option `UCI_ShowWDL`, oubliée au premier essai —, erreurs par source, bootstrap par partie, verdict). Chiffres sans valeur de comparaison : le réseau B avait été entraîné sur ces mêmes positions (erreur 0,015 contre leurs résultats = mémorisation). Le vrai A/B se fera sur Kaggle avec des parties réellement mises de côté.

## Étape 6e — Notebook Kaggle de self-play (2026-10-22)

### Décisions
- **Ancres fixes pour l'étape 7** (proposées par Claude, validées) : arène aussi contre le réseau supervisé, et protocole v2 contre Stockfish toutes les ~5 générations.
- **Réseau de départ : Dataset plutôt que sortie du notebook 4b.** Ajouter la sortie du notebook 4b comme entrée marche aussi (*Add Input → Your Work → ce notebook*), mais l'entrée suit une version de ce notebook : si on le relance un jour (essai, débogage), sa sortie change. Un Dataset est figé, versionné, et servira d'ancre fixe à l'étape 7. Dans les deux cas, le notebook vérifie le SHA-256 (`c1f616c6…18b33`) et prend le premier `.pt` d'entrée conforme.
- **Essai de débit en deux phases.** Toutes les parties d'un processus avancent ensemble : une partie dure environ (parties en parallèle × 200 simulations × ~93 coups cherchés) / débit du processus, soit ~15 min sur Kaggle avec 64 parties. Un essai de 15 min avec les vrais réglages n'aurait presque aucune partie terminée. Phase 1 (5 min, vrais réglages) pour le débit, le GPU, le CPU et la taille des lots ; phase 2 (10 min, 16 parties par processus, ~4 min par partie) pour les longueurs, les fins et les répétitions. Parties/h = (simulations/s de la phase 1 / 200) / coups cherchés par partie de la phase 2. Production de positions = simulations/s / 200 × 3 600 (une position enregistrée par coup cherché).
- **Dépôt et tables dans `/tmp`** : pas dans `/kaggle/working`, donc pas dans la sortie de la session (les tables feraient 396 Mo de plus à chaque version, retéléchargées en 1 à 2 min de toute façon).

### Code
- `scripts/download_syzygy.py` : liste officielle (`sha256` du miroir Lichess), téléchargement parallèle, vérification de chaque fichier, fichiers conformes déjà présents gardés, 3 essais, code de sortie 1 si un fichier reste absent ou non conforme. Vérifié : 145 fichiers WDL conformes trouvés dans `tools/syzygy` (396,1 Mo) ; téléchargement réel d'un fichier ; un fichier à mauvaise somme est rejeté et supprimé.
- `scripts/selfplay_session.py` : `bench` (deux phases, `ResourceSampler` = `nvidia-smi` + `/proc/stat` toutes les 2 s, `check_selfplay.py` sur la phase 2, rapport, `bench_report.json`) ; `full` (tampon repris par `--inputs`, self-play seul par défaut ; `--with-training` = boucle simple provisoire : tranches de `--chunk-minutes` (120) de self-play avec le dernier réseau accepté, puis un tour de `train_selfplay.py`, arrêt sur G4). Rapport : débit, lots envoyés au GPU, parties, répétitions, estimation des heures restantes pour 250 000 positions d'entraînement.
- `scripts/selfplay.py` : marges de fin réglables (`--stop-new-games-minutes`, `--hard-stop-minutes`) ; fichier de session enrichi (temps de jeu, simulations, positions par appel au réseau, part du réseau dans le temps, longueur de chaque partie) ; les processus fils écrivent en UTF-8.
- `notebooks/kaggle_selfplay.ipynb` (8 cellules) : 0 réglages (`MODE`, `BENCH_MINUTES` 15, `FULL_MINUTES` 660, `SELFPLAY_ONLY`, `WORKERS`, `GAMES_PER_WORKER`, SHA-256 attendu) ; 1 clone dans `/tmp` + `pip install -e` sans toucher à torch ; 2 environnement (GPU obligatoire, cœurs → 3 processus) ; 3 entrées (réseau par SHA-256, Lichess, tampon et état d'entraînement des sessions précédentes) ; 4 tables WDL ; 5 `selfplay_session.py` ; 6 contenu de la sortie.

### Tests locaux (CPU : les chiffres de débit n'ont aucun sens pour Kaggle, seule la mécanique compte)
- `selfplay_session.py bench` (2 processus × 2 threads, phases de 2 et 5 min) : rapport complet (débit, lots de 16, parties de la phase 2, parties/h estimées, heures pour 250 000 positions). Un premier essai en une seule phase n'avait **aucune partie terminée** en 3 min : c'est ce qui a conduit aux deux phases.
- `selfplay_session.py full` : session A (self-play seul, 77 positions) puis session B avec `--inputs` = sortie de A (1 shard copié, 306 positions au total) et `--with-training` : « pas de tour, tampon de 306 positions, minimum 250 000 ».
- Cellules 3 à 6 du notebook exécutées dans un faux `/kaggle` : réseau trouvé par son SHA-256 dans un faux Dataset, tampon précédent trouvé, commande de session complète correcte ; en mode essai, la chaîne complète tourne (tables vérifiées, deux phases, rapport). Observé pendant cet essai : le budget n'est vérifié qu'entre deux itérations (une itération = toutes les parties d'un processus avancent d'un coup) ; sur CPU une itération de 64 parties dure plusieurs minutes (phase 1 de 8,3 min pour 5), sur Kaggle ~10 s (dépassement négligeable). Ce test a aussi tourné en partie en même temps que les processus d'un premier essai interrompu (erreur d'affichage d'un caractère sous Windows, corrigée), d'où un débit anormalement bas : sans conséquence sur la mécanique vérifiée.

### Instructions Kaggle pas à pas
1. **Réseau de départ en Dataset** (une seule fois) : kaggle.com → *Create* → *New Dataset* → glisser `checkpoints\kaggle_sup\best.pt` (37,8 Mo) → titre `chesszero-kaggle-sup` → *Private* → *Create*. (Sinon : *Add Input → Your Work → Notebooks →* le notebook de l'étape 4b ; le SHA-256 garantit que c'est le bon fichier.)
2. **Importer le notebook** : *Create* → *New Notebook* → *File* → *Import Notebook* → onglet *GitHub* (ou *Link*) avec `https://github.com/Enz091/Project_chess/blob/main/notebooks/kaggle_selfplay.ipynb`, ou téléverser le fichier `notebooks\kaggle_selfplay.ipynb` du dépôt local.
3. **Réglages de la session** (panneau de droite, *Session options*) : *Accelerator* = **GPU T4 x2** (ou P100) ; *Internet* = **On** ; *Persistence* = *No persistence*.
4. **Entrées** (*Add Input*) : *Datasets* → `chesszero-kaggle-sup` ; *Datasets* → `lichess-24m` (inutile en self-play seul, sans danger).
5. **Essai de débit** : laisser `MODE = "bench"` → *Run All* (environ 20 min avec l'installation et les tables) → lire le rapport de la cellule 5 → arrêter la session (le quota GPU court tant qu'elle est ouverte).
6. **Session complète de self-play** : cellule 0 → `MODE = "full"`, `SELFPLAY_ONLY = True` → *Save Version* → *Save & Run All (Commit)*. La version tourne en arrière-plan jusqu’à 11 h environ (`FULL_MINUTES = 660`, 1 h de marge sous la limite de 12 h) et consomme ~11 h de quota GPU. Sortie : onglet *Output* de la version : `selfplay/` (tampon) et `selfplay/sessions/<session>.json` (statistiques, dont les répétitions précoces).
7. **Session suivante** : ouvrir le notebook → *Add Input → Your Work → Notebooks →* ce notebook (vérifier que c'est la dernière version terminée) → *Save & Run All (Commit)*. La cellule 3 trouve `selfplay/` dans `/kaggle/input/...` ; la nouvelle sortie contient à nouveau toute la fenêtre du tampon (500 000 positions au plus).

### Correction (2026-10-22, demande de l'auteur) : budget de la session complète 690 → 660 min
Une version Kaggle coupée par la limite des 12 h peut perdre toute sa sortie (donc le tampon). Avec 690 min, il ne restait que 30 min pour l'installation (clone, `pip`, tables : 3 à 5 min), le dépassement possible du budget (une itération, ~10 s), la synchronisation finale du tampon et la sauvegarde de la sortie. **`FULL_MINUTES = 660` par défaut** (11 h) : environ 1 h de marge. Rappel : à l'intérieur de ce budget, `selfplay.py` ne démarre plus de parties 20 min avant la fin et s'arrête 5 min avant. Vérification au même moment : aucun processus orphelin des essais (Python, Stockfish, Cute Chess, ni aucun processus mentionnant le projet) ne tournait sur le PC.

## Étape 7 — Plan de la boucle AlphaZero sur Kaggle (validé par l'auteur le 2026-10-22, pas encore codé)

Plan proposé par Claude, validé avec des changements de l'auteur (marqués « auteur »). Chiffres de départ : débit mesuré sur Kaggle **3 900 simulations/s** (3 processus) → **70 200 positions/h** (3 900 / 200 × 3 600), ~750 parties/h (~93 coups cherchés par partie).

### Changement de sens : candidat puis arène
Aujourd'hui `train_selfplay.py` « accepte » une génération dès que les garde-fous passent. À l'étape 7, un tour qui passe les garde-fous produit un **candidat** ; c'est l'**arène** qui l'accepte ou le rejette. Les compteurs de génération (passage à 10 % d'humain dès la 6e génération acceptée, G1 / G2 stricts pendant les 5 premières) comptent désormais les générations **acceptées en arène**.

### Arène
- **Ouvertures** : réserve d'arène d'environ 2 000 positions du livre, choisies de façon stable par `crc32(EPD)`, que le self-play ne tire plus à partir de l'étape 7 (la session de self-play en cours en aura touché 1 à 2 % : négligeable, l'arène mesure la force et non la mémorisation). Les 40 ouvertures v2 sont déjà exclues du livre (transpositions comprises, `tests/test_book.py`).
  - **Arène A** (candidat contre réseau accepté) : 100 positions de la réserve, graine = numéro du candidat (nouvelles à chaque arène), chacune jouée deux fois en inversant les couleurs = **200 parties**.
  - **Arène ancre** (candidat contre `kaggle_sup`) : 100 positions **fixes** (les mêmes à chaque génération), deux couleurs = 200 parties ; comparaisons appariées d'une génération à l'autre. Jouée **seulement si l'arène A est gagnée**, et pas tant que le réseau accepté est `kaggle_sup` lui-même (génération 1 : les deux arènes seraient identiques).
- **Conditions** : **200 simulations par coup** (= self-play et protocole v2), température 0, pas de Dirichlet, un arbre par réseau réutilisé d'un coup à l'autre, arbitrage par les tables WDL dès 5 pièces, nulle à 400 demi-coups, **abandon seulement si les deux camps sont d'accord** (perdant Q < −0,90 et gagnant Q > +0,90 pendant 3 coups). Aucune position de l'arène n'entre dans le tampon. Dans chaque processus, les parties où le candidat est au trait partent à son réseau, les autres à l'autre réseau (deux appels de demi-taille par itération ; débit à peu près inchangé, le goulot étant le CPU).
- **Règle (validée)** : acceptation si **score ≥ 52 %** (≥ 104 points sur 200, ≈ +14 Elo). Intervalle de l'Elo calculé **par paires d'ouvertures** ; le rapport donne aussi la probabilité que le candidat soit meilleur. Probabilité d'accepter selon le vrai écart (200 parties, 30 % de nulles) :

  | Seuil | −40 | −20 | 0 | +20 | +40 | +60 |
  |---|---|---|---|---|---|---|
  | ≥ 50 % | 3 % | 17 % | 50 % | 83 % | 97 % | 100 % |
  | **≥ 52 %** | 0 % | 5 % | 25 % | 62 % | 90 % | 99 % |
  | ≥ 55 % (AlphaGo Zero) | 0 % | 0 % | 5 % | 24 % | 60 % | 88 % |

  Intervalle à 95 % d'une arène : ±35 à 40 Elo à 200 parties (±25 à 29 à 400).
- **Comparaison** : AlphaGo Zero jouait 400 parties à 1 600 simulations et acceptait au-delà de 55 %, mais son entraînement continuait sans interruption (un rejet retardait seulement la mise à jour du joueur de self-play). AlphaZero (comme Leela et KataGo) n'avait pas d'arène : un réseau mis à jour en continu, le self-play prend toujours le dernier. À notre échelle, 55 % refuserait les trois quarts des vrais gains de +20 Elo, taille probable des gains par génération.
- **Pas de retour en arrière après un rejet en arène (validé)** : la ligne d'entraînement garde ses poids et son optimiseur, les positions du tour sont marquées comme utilisées, le candidat suivant affronte le même réseau accepté ; seul le self-play reste sur l'ancien réseau. Un rejet par les garde-fous ramène toujours au dernier état sain (`line_ok.pt`, l'actuel `accepted.pt` renommé).
- **Veto de l'ancre (validé)** : candidat refusé si son Elo contre `kaggle_sup` est inférieur à celui du réseau accepté de plus de 2 erreurs types de l'écart (~57 Elo) : rattrape un progrès non transitif.
- **Coût** : 200 parties × ~92 coups cherchés × 200 simulations ≈ 3,7 M de simulations ≈ 16 min de calcul, **~19 min de quota** avec la fin des longues parties et le chargement. Candidat rejeté en arène A : ~19 min ; candidat accepté : ~38 min (deux arènes), sauf en génération 1.
- **Contrôles de l'arène (auteur)**, en local en version réduite puis une fois au début de la première session Kaggle : (a) `kaggle_sup` contre lui-même → score ~50 % ; (b) `kaggle_sup` à 200 simulations contre `kaggle_sup` à 50 simulations → le côté à 200 doit gagner largement. But : détecter une erreur d'attribution des couleurs ou des réseaux.

### Arrêts pour décision humaine (révisés par l'auteur)
- **Stagnation : 5 rejets en arène d'affilée** (et non 3 : avec ~14 candidats par semaine, 3 rejets d'affilée arriveraient par simple bruit dans 40 à 80 % des semaines même si l'apprentissage marche).
- **Ancre : réseau accepté sous `kaggle_sup` (borne haute de l'intervalle < 0) sur 2 générations acceptées consécutives, à partir de la génération 3** (une mesure isolée à ±38 Elo peut déclencher à tort).
- G4 inchangé (3 rejets d'affilée par les garde-fous).
- Un arrêt écrit le rapport et **termine la session** (économie de quota).

### Boucle dans une session
- Orchestrateur `scripts/alphazero_session.py`, état dans `loop/loop_state.json`. Une génération : (1) self-play avec le réseau accepté jusqu'à **`new_positions_per_generation` nouvelles positions (paramètre de configuration, 100 000 par défaut ≈ 85 min, 521 pas ; l'auteur l'ajustera, probablement 150 000 à 200 000, selon le taux d'acceptation)** ; (2) tour d'entraînement (~5 à 8 min avec les évaluations) ; (3) garde-fous G1-G3 : échec → rejet, retour à (1) ; (4) arène A : défaite → rejet en arène, la ligne continue, retour à (1) ; (5) arène ancre si nécessaire (veto possible) ; (6) acceptation → `gen_XXXX.pt`, le self-play reprend avec ce réseau.
- **Self-play pendant l'entraînement et l'arène : arrêt puis relance (code actuel), décision de l'auteur** ; la pause avec changement de réseau à chaud (ancienne sous-étape 7c, gain estimé ~5 % du temps) est **reportée**. Faire tourner les deux en même temps n'apporterait rien : 4 cœurs, déjà le goulot, et l'arène fait le même travail que le self-play.
- **Lichess** : sous-ensemble fixe de 6 M de positions (graine fixe : lots identiques à la reprise) chargé une fois par session dans un cache `/tmp`, au lieu des 24 M à chaque tour (un tour n'en utilise que ~133 000).
- **Temps (budget ≤ 660 min)** : ~10 min de démarrage. Un tour ne démarre que s'il reste ≥ entraînement + arène A + arène ancre estimés + 15 min (durées mesurées des sessions précédentes ; au départ 8, 20 et 20 min) ; sinon self-play jusqu'aux marges de fin existantes (pas de nouvelles parties 20 min avant, arrêt 5 min avant), les positions en trop vont au tour suivant. Une arène ne démarre que s'il reste sa durée estimée + 15 min ; sinon le candidat attend et la session suivante commence par lui. Fin pendant un entraînement : arrêt propre avec `train_state.pt`, reprise au pas près. Fin pendant une arène : parties terminées dans `arena/<id>/games.jsonl`, la reprise ne rejoue que les couples (ouverture, couleur) manquants. Coupure Kaggle à 12 h : on repart de la sortie précédente (au plus une session perdue).

### Reprise d'une session à l'autre
- **Sortie de chaque session** : `selfplay/` (tampon, `sessions/*.json`) ; `sp_train/train_state.pt` (ligne d'entraînement : poids, optimiseur, pas global, `lr_scale`, `reuse`, `human_sticky_25`, shards utilisés, historique des garde-fous, compteurs) ; `sp_train/line_ok.pt` ; `sp_train/candidate.pt` (candidat en attente et ses métriques) ; `sp_train/generations/gen_XXXX.pt` (réseaux acceptés en arène, format `load_for_inference`) ; `loop/loop_state.json` (numéro de session, phase, génération acceptée, numéro du candidat, rejets d'affilée, Elo contre l'ancre par génération, durées des phases) ; `loop/arena/history.jsonl`, `loop/arena/<id>/games.jsonl` + `.pgn` ; `loop/reports/`.
- **Ordre d'écriture (auteur)** : fichiers de données (`gen_XXXX.pt`, `candidate.pt`, `train_state.pt`, `line_ok.pt`) écrits de façon atomique (fichier temporaire puis `os.replace`), **`loop/loop_state.json` écrit en dernier** comme point de validation. La reprise tolère un **fichier orphelin** (écrit mais non référencé par `loop_state.json`) et une **dernière ligne tronquée** dans `arena/<id>/games.jsonl`. Tests de coupure pour ces deux cas.
- **Reproductibilité (auteur)** : chaque entrée de `loop/arena/history.jsonl` contient le hash git du code et un instantané de la configuration.
- **Lancement** : même notebook `kaggle_selfplay.ipynb`, nouveau `MODE = "loop"` (réimporter une fois le notebook dans Kaggle) ; entrées : `chesszero-kaggle-sup`, `lichess-24m` (obligatoire), sortie de la dernière session terminée de ce notebook (retirer l'ancienne) ; GPU T4 x2 ou P100, Internet activé ; `FULL_MINUTES` réglable. La cellule 3 affiche « reprise de la session n°k, génération g, candidat c en attente » et refuse deux états ou un état sans tampon. Une seule session à la fois.
- **Première session en `MODE = "loop"` (auteur)** : courte, ~3 à 4 h, pour valider la chaîne réelle (entrées, contrôles de l'arène, une arène, reprise). Les sessions de 11 h ensuite.

### Suivi et décisions
- **Rapport** `loop/reports/session_XXXX.md` (affiché aussi en fin de notebook) : « Actions pour toi » en tête ; par candidat : pas, pertes, validation Lichess (contre le supervisé et la génération précédente), erreur de valeur des parties mises de côté, garde-fous, arène A (V/N/D, score, Elo ± intervalle, probabilité d'être meilleur, décision), arène ancre ; self-play : parties, positions, simulations/s, GPU/CPU, fins, nulles par répétition (self-play et arène), faux abandons ; état final ; chronologie des phases. Cumul : `loop/history.csv` et `loop/progress.png` (Elo contre l'ancre par génération).
- **Mesure v2 contre Stockfish (auteur)** : « Actions pour toi » la recommande quand l'Elo contre l'ancre, lissé sur les dernières générations, a gagné plus de +50 depuis la dernière mesure, ou au plus tard toutes les ~10 générations acceptées. Sur le PC : télécharger `gen_XXXX.pt`, `match_stockfish.py run --tag sp_genXXXX --checkpoint <gen_XXXX.pt> --syzygy tools\syzygy --like kaggle_sup` (~2 h 20, copie figée, PC sans autre charge), puis `compare kaggle_sup sp_genXXXX` (écart < ~55 Elo non significatif).
- **A/B de la cible de valeur (validé)** : sur le PC juste après la session de self-play en cours, sans bloquer la boucle (qui démarre en `result`) : télécharger `selfplay/`, `eval_value_target.py label` (Stockfish sur l'i9), un tour `result` et un tour `blend` depuis le réseau supervisé sur CPU (~325 pas chacun, durée à mesurer et annoncer), `compare`. Si `blend` gagne sur les deux mesures, changement au début d'un tour.
- **Outil en ligne de commande Kaggle (7f, facultatif)** : `kaggle kernels output <compte>/<notebook> -p <dossier>` récupère la sortie de la dernière version (options exactes à vérifier avec `--help`). L'auteur installe le paquet `kaggle` dans le venv et crée lui-même le jeton (*Settings → API*), placé dans `C:\Users\Asus\.kaggle\` ou en variable d'environnement. **Le jeton ne doit jamais entrer dans le dépôt** : `.gitignore` couvre `kaggle.json`, `.kaggle/` et `access_token` (ajoutés le 2026-10-22, vérifié avec `git check-ignore`).

### Quota (estimation)
Cycle par candidat ≈ 85 min de self-play + 6 min d'entraînement + 19 min d'arène A (+ 19 min d'arène ancre si accepté) ≈ 110 à 130 min. 30 h par semaine (deux sessions de 11 h + une d'environ 7 h ≈ 1 700 min de boucle) → **~13 à 14 candidats, 6 à 9 générations acceptées par semaine** à 100 000 positions par génération ; ~1,4 M de positions, ~15 000 parties. Sessions longues, une à la fois. À vérifier : décompte de T4 x2 dans le quota (le second GPU ne sert pas, le CPU est le goulot).

### Réserves
Échelle (15 000 parties par semaine, contre des millions pour AlphaGo Zero et AlphaZero) : gains par génération souvent sous la résolution d'une arène, progrès visible surtout dans la tendance contre l'ancre et les mesures v2. Départ d'un réseau supervisé solide : régression possible des premières générations, plateau réaliste ; estimation sans garantie de 0 à +150 Elo (échelle v2) en quelques semaines si tout va bien. Indices après 3 à 5 générations : erreur de valeur des parties mises de côté en baisse, taux d'acceptation > 40 %, Elo contre l'ancre en hausse. Bruit du filtrage (~5 % des régressions de −20 Elo passent), continuation sur des données d'un réseau en retard, rétrécissement des données (répétitions, température sur 30 demi-coups seulement), chaîne manuelle des entrées Kaggle.

### Sous-étapes (ordre validé)
- **7a** : `src/chesszero/arena.py` + `scripts/arena.py` (plusieurs processus, reprise, Elo par paires), réserve d'ouvertures exclue du self-play, contrôles (a) et (b).
- **7b** : candidat / acceptation dans `selfplay_train.py` (continuation après rejet en arène, compteurs de générations acceptées, `line_ok.pt`, `candidate.pt`).
- **7d** : orchestrateur, budget de temps, arrêts, ordre d'écriture et tolérance aux coupures, rapport.
- **7e** : `MODE = "loop"` dans le notebook, essai complet en local, puis première session Kaggle courte.
- **7f** (facultatif) : `scripts/fetch_kaggle.py`.
- (Reportée : pause du self-play avec changement de réseau à chaud.)

## Étape 7a — Arène et réserve d'ouvertures (2026-10-22)

Vérification préalable (demande de l'auteur) : `git log --all --oneline -- "*kaggle.json" "*access_token" ".kaggle/*"` vide, et aucun de ces fichiers dans l'arbre d'aucun des 26 commits. Commit séparé du plan et du `.gitignore` (`fe7825a`), poussé.

### Réserve d'arène (écart d'exécution par rapport au plan)
- La règle est celle du plan : une position du livre est réservée si `crc32("arena:" + EPD) % 1000 < 9` → **1 991 positions sur 225 945 (0,88 %)**.
- **Écart** : rejouer les 226 000 suites pour leur EPD prend 40 à 70 s, à chaque chargement du livre par chaque processus de self-play (sans doute 2 fois plus sur Kaggle). La réserve est donc **calculée une fois et versionnée** : `books/lichess_2000_arena.tsv.gz` (37 Ko, même format que le livre) + `books/lichess_2000_arena.json` (règle, nombre, SHA-256 du livre). `prepare_book.py --arena-reserve` la recalcule (et elle est recalculée après une extraction complète ou `--refilter`).
- Self-play : `exclude_arena_reserve` retire ces suites du tirage (appartenance d'une chaîne, instantané). **Les indices de départ restent ceux du livre complet** (`OpeningBook.allowed` + `sample_index`), donc `check_selfplay.py` et les shards déjà écrits restent cohérents. Le nombre d'entrées exclues est noté dans les métadonnées des shards (`arena_reserve_excluded`). La session de self-play en cours sur Kaggle (code d'avant) n'est pas concernée.

### Code
- `src/chesszero/arena.py` :
  - `ArenaConfig` (section `[arena]` de `kaggle.toml`), `Player` ;
  - `anchor_openings` (les 100 premières d'un ordre fixe de la réserve), `gate_openings` (100 tirées hors ancre, graine = numéro du candidat), `schedule` (chaque ouverture avec les deux couleurs) ;
  - `ArenaGame` + `play_games` : un arbre par joueur, les arbres des parties où A est au trait partent au réseau A en un appel, ceux de B au réseau B ; coups forcés sans recherche ; abandon seulement si les deux camps sont d'accord (Q racine et Q du meilleur coup, des deux côtés, sur leurs `resign_consecutive` derniers coups) ;
  - `summarize` : score, V/N/D, **erreur type par paires d'ouvertures**, Elo et intervalle à 95 %, probabilité que A soit meilleur, fins, répétitions avant le demi-coup 60, score des Blancs, paires à exactement 1 point sur 2 ;
  - `gate_decision` : acceptation à 52 %, veto de l'ancre à 2 erreurs types de l'écart (les règles d'arrêt « 5 rejets » et « 2 générations sous l'ancre » viendront avec l'orchestrateur, 7d) ;
  - fichiers : `arena.json` (joueurs identifiés par le **SHA-256** de leur checkpoint, car les chemins changent d'une session Kaggle à l'autre ; ouvertures, réglages, code git), `games.jsonl`, `games.pgn`, `result.json` (en dernier) ; `append_history` (une seule ligne par arène, avec hash git et instantané de la configuration) ; `process_runner` (processus en `spawn`, seul le processus principal écrit `games.jsonl`, les fils lui envoient leurs parties par une file).
- `scripts/arena.py` : `--kind gate|anchor`, `--seed`, `--pairs`, `--a/--b` (+ `--a-sims/--b-sims`, noms), `--max-minutes`, `--stop-after`, `--history`, `--expect-min/--expect-max` (contrôles) ; relancer = reprendre ; codes de sortie 0 complet, 1 contrôle échoué, 3 incomplet.
- `scripts/prepare_book.py --arena-reserve`, `book.py` (règle, chemin, calcul, exclusion), `selfplay.py` (tirage par `sample_index`).

### Tests (`tests/test_arena.py`, 15 ; suite complète : 223 tests en 3 min 14)
- Ouvertures de l'ancre fixes, ouvertures de l'arène contre l'accepté reproductibles, dépendantes de la graine, jamais dans l'ancre ; deux couleurs par ouverture.
- Réserve versionnée conforme à la règle (toutes ses entrées, et un échantillon de 20 000 entrées du livre hors réserve) ; le self-play ne tire jamais la réserve et garde les indices du livre complet.
- **Attribution des couleurs** : départ où les Blancs matent en un coup → A gagne avec les Blancs, perd avec les Noirs. **Attribution des réseaux et des simulations** : A à 40 simulations, B à 8 → chaque faux réseau évalue ~ses propres simulations par coup de SON joueur. **Abandon** : déclenché quand les deux camps sont d'accord, jamais quand l'adversaire ne se voit pas gagnant.
- Erreur type par paires, Elo, probabilité d'être meilleur, 52 % ≈ +13,9 Elo ; 104/200 accepté, 103 refusé ; veto de l'ancre juste au-delà de 2 erreurs types, pas juste en deçà.
- **Coupures** : dernière ligne tronquée de `games.jsonl` ignorée puis retirée (une ligne complète illisible = erreur) ; arène coupée après 2 parties, puis ligne tronquée et `result.json.tmp` / `games.pgn.tmp` orphelins → la reprise ne rejoue que les 4 parties manquantes et donne **exactement** le résultat d'une arène d'une traite ; arène terminée jamais rejouée ; arène différente (simulations, ouvertures) refusée ; historique sans doublon avec hash git et configuration ; chemin réel à plusieurs processus avec un petit réseau (coupure puis reprise).
- **Reste à couvrir en 7d** : le fichier orphelin au sens de l'orchestrateur (`gen_XXXX.pt` ou `candidate.pt` écrit mais non référencé par `loop_state.json`), qui n'existe pas encore en 7a.

### Diagnostic du débit sur le PC (demande de l'auteur)
Première mesure de l'arène en local : ~340 simulations/s, environ deux fois moins que l'essai de self-play de l'étape 6c (648). Comparaison dans les mêmes conditions (réseau kaggle, CPU, 1,5 min chacun, simulations/s rapportées au temps de jeu) :

| Disposition | Self-play | Arène (même réseau des deux côtés) | Écart |
|---|---|---|---|
| 2 processus × 2 threads, 4 parties par processus (conditions de 6c) | **447** (16 positions par appel, réseau 92 % du temps) | **403** (8 par appel et par joueur) | −10 % |
| 2 processus × 4 threads, 10 parties par processus | 401 (39 par appel, réseau 87 %) | 327 | −18 % |

- **Cause principale : la machine.** Le même self-play qu'en 6c fait 447 simulations/s aujourd'hui, contre 648. Le PC tournait **sur batterie** (statut « en décharge », 37 %, mode d'alimentation « Utilisation normale »), ce qui bride très probablement le processeur du portable (cause probable, non prouvée par une mesure sur secteur). Indice concordant : 4 threads par processus vont moins vite que 2.
- **Part de l'arène : −10 à −18 % sur CPU.** Dans l'arène, chaque itération fait deux appels au réseau de demi-taille (un par joueur) au lieu d'un : sur CPU, où le réseau prend ~90 % du temps, des lots plus petits sont moins efficaces.
- **Impact sur Kaggle : faible, l'estimation de ~19 min par arène tient.** Sur Kaggle, le goulot est la recherche en Python (CPU) et le GPU est peu chargé : deux lots de ~128 positions au lieu d'un de ~256 ajoutent surtout la latence d'un appel GPU par tour de collecte (~50 tours de 4 feuilles par coup, quelques ms chacun), soit de l'ordre de 2 à 5 % sur une itération d'~10 s. Compter ~19 à 20 min ; à vérifier sur la première arène Kaggle (le rapport donne les simulations/s). Aucun changement de code.

### Contrôles de l'arène (version réduite, PC, réseau `kaggle_sup`, 2 processus × 2 threads, PC sur batterie)
- **(a) `kaggle_sup` contre lui-même, 200 simulations, 5 ouvertures fixes de l'ancre (10 parties), 9,5 min** : 4 victoires, 2 nulles, 4 défaites, **score exactement 50,0 %**, **5 paires sur 5 à exactement 1 point** (chaque ouverture donne le même résultat vu des Blancs dans les deux parties : jeu déterministe) ; score des Blancs 70 % ; fins : 6 abandons (d'accord des deux camps), 2 mats, 2 répétitions ; 98 demi-coups en moyenne. CONTRÔLE RÉUSSI. L'intervalle affiché est « −0 à −0 » : erreur type nulle quand toutes les paires sont identiques (affichage sans conséquence).
  - **Limite (remarque de l'auteur)** : avec deux joueurs identiques, ce contrôle **ne peut pas détecter une inversion des réseaux ou des couleurs**. Il vérifie la **comptabilité des paires** (deux parties par ouverture, couleurs alternées, points comptés pour le bon joueur dans chaque partie : une erreur qui donnerait par exemple toujours à A le résultat des Blancs aurait donné ~70 % ici). L'attribution elle-même est vérifiée par le contrôle (b) et par les tests unitaires (mat en un coup, simulations par joueur).
- **(b) `kaggle_sup` à 200 simulations contre `kaggle_sup` à 50, 10 ouvertures (20 parties), 13,3 min** : 17 victoires, 2 nulles, 1 défaite, **score 90,0 %, Elo +382 (intervalle à 95 % : +232 à +1200, borne haute plafonnée)**, probabilité d'être meilleur ~100 % ; fins : 12 abandons, 4 mats, 2 arbitrages par les tables, 2 répétitions ; 81 demi-coups en moyenne ; score des Blancs 50 %. CONTRÔLE RÉUSSI (seuil 70 %). Cohérent avec le match de vérification de l'étape 6 (64 contre 200 simulations : −421 ± 139 Elo).
- Débit : 287 simulations/s en (a), 205 en (b) (le côté à 50 simulations fait des lots plus petits ; fin des dernières parties). Durées conformes à l'estimation (10 à 12 min et 12 à 15 min).
- Parties : `docs/arena_controle_a.pgn`, `docs/arena_controle_b.pgn` ; résultats complets dans `data/arena_controle_a|b/` (hors git). `result.json` porte le commit `fe7825a` avec `dirty = true` (code de 7a pas encore commité).
- **À refaire au début de la première session Kaggle en `MODE = "loop"`** (plan validé) : mêmes deux contrôles sur GPU, avant la première vraie arène.

### Alimentation et mesures (2026-10-22, après le commit de 7a)
- **Règle ajoutée dans CLAUDE.md** (demande de l'auteur) : mesures v2 et diagnostics de débit sur secteur, en mode « meilleures performances ». Sur batterie, Stockfish (300 ms par coup) s'affaiblit et l'IA (simulations fixes) non : l'Elo serait gonflé.
- **La mesure de référence 2427 ± 37 ne note pas l'alimentation** (ni le journal, ni le bloc `environment` de v1 / v2). Vérification indirecte : `match_stockfish.py bench --bench-games 16 --syzygy tools/syzygy` **sur secteur, mode « meilleures performances »** : 0,86 s par coup de l'IA (0,83 s sans les 4 premières parties, qui ouvrent les tables), 52 s par partie, ~140 min estimées pour 640 parties ; **référence v2 : 0,80 s par coup, 53 s par partie, 142 min** (v1 : 0,81 s). Mêmes conditions de vitesse : la référence a très probablement été jouée sur secteur à pleine vitesse (sur batterie, le débit tombait à ~65 % : le coup aurait pris ~1,2 s et la mesure ~3 h 30).
- Mode d'alimentation de Windows 11 : c'est une surcouche du plan (`powercfg` affiche toujours « Utilisation normale ») ; lue dans `HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes` : sur secteur `ded574b5…` = meilleures performances ; **sur batterie `961cc777…` = meilleure efficacité énergétique**, d'où le bridage observé pendant le diagnostic de 7a. La suite de tests prend 1 min 46 sur secteur contre 3 min 14 sur batterie.
- `match_stockfish.py` : bloc `environment` + `power` (secteur ou batterie, charge, plan, mode d'alimentation et son GUID), et un **bandeau d'avertissement** au lancement de `run` et de `bench` si la mesure part sur batterie, hors « meilleures performances », ou si l'alimentation est inconnue (test dans `tests/test_ladder.py`).

### Relecture statistique de l'arène (corrections, commit `74e834c`)
Cas limites reproduits sur le code de 7a, puis corrigés (un test par cas dans `tests/test_arena.py`) :
- **Erreur type nulle** (paires toutes identiques comme au contrôle (a), uniquement des nulles, 100 %, 0 %) : intervalle « −0 à −0 » et surtout **veto de l'ancre déclenché par un écart de −1 Elo** (comparaison à 2 × 0). Correction : erreur type au moins égale à une demi-partie (0,5 / n), le score n'étant pas connu plus finement ; à 200 parties ce plancher vaut ~1,7 Elo et ne change rien aux vraies arènes. Le contrôle (a) affiche maintenant « +0 (−69 à +69) » sur 10 parties.
- **Une seule paire ou une seule partie** : erreur type NaN écrite dans `result.json` (JSON non conforme) et traitée comme 0 par le veto. Correction : toujours des nombres finis (vérifié en JSON strict) ; une erreur type manquante est une erreur, pas un veto.
- **Paires incomplètes** (arène reprise) : estimateur par grappes (`score_se`), identique à l'ancien pour des paires complètes (testé), juste pour des paires inégales.
- **100 % ou 0 %** : Elo calculé avec une demi-partie de moins, marqué comme borne (« ≥ » / « ≤ ») ; borne d'intervalle « non bornée » au lieu du plafond ±1200 affiché comme une mesure (le « +1200 » du contrôle (b) était ce plafond).
- **Arène incomplète** : `gate_decision` refuse de décider.
- Inchangé : Elo et intervalle calculés en score puis convertis ; probabilité d'être meilleur Φ((score − 0,5) / erreur type) ; seuil de 52 % sur le score seul (100 % accepté, 0 % et nulles seules rejetés). Le bruit de la mesure de l'accepté contre l'ancre est pris en compte (racine de la somme des carrés des deux erreurs) ; les deux arènes ancre se jouent sur les mêmes ouvertures, leurs erreurs sont plutôt corrélées positivement, donc l'erreur de l'écart est un peu surestimée et le veto est prudent (une version appariée par ouverture serait plus puissante : non retenue, elle changerait la règle validée).

## Étape 7b — Candidat puis arène dans l'entraînement (2026-10-22)

### Conception validée par l'auteur
- Un tour qui passe les garde-fous donne un **candidat** (`candidate.pt`) ; `commit_arena` applique le verdict de l'arène. **Rejet en arène : la ligne d'entraînement continue** (rien n'est défait, positions marquées comme utilisées) ; rejet par les garde-fous : retour à `line_ok.pt` (dernier état qui a passé les garde-fous, l'ancien `accepted.pt`). **Pas de baisse du taux après un rejet en arène.** Chute brutale comparée au dernier état de la ligne (`previous` = métriques du dernier état qui a passé les garde-fous).
- Générations comptées = **acceptées en arène** : G1 / G2 stricts pendant les 5 premières, 10 % d'humain dès la 6e.
- Arrêts : **5 rejets en arène d'affilée** (les rejets par les garde-fous n'y comptent pas et ne remettent pas le compteur à zéro ; G4 garde son propre compteur) ; **réseau accepté sous `kaggle_sup` (borne haute < 0) sur 2 générations acceptées consécutives, en ne comptant que les générations ≥ 3 (lecture A : premier arrêt possible à la génération 4)**. Un arrêt bloque les tours ; `train_selfplay.py --clear-stop "note"` repart sur décision humaine (remet à zéro le compteur concerné ; pour l'ancre, il faudra deux nouvelles générations sous l'ancre).
- **Génération 1 (remarque de l'auteur)** : tant que le réseau accepté est le réseau de départ (= l'ancre), la même arène sert à l'acceptation et à la mesure contre l'ancre ; cette mesure est **gonflée par la sélection (estimation de l'auteur : 10 à 20 Elo)**. Choix gardé ; la source est enregistrée (`anchor_source` dans `train_state.pt` et dans `gen_0001.pt`). **Le rapport de la génération 2 (7d) devra le rappeler** : un veto de l'ancre à la génération 2 compare le candidat à une référence un peu surestimée, donc plus sévère que prévu.
- **Source de vérité unique** : `train_state.pt` pour le candidat en attente, la génération acceptée, les rejets en arène, l'historique contre l'ancre et l'arrêt ; `read_train_state(dossier)` est la fonction de lecture que l'orchestrateur (7d) utilisera, sans recopier ces informations dans `loop_state.json`.

### Code
- `selfplay_train.py` : état au **format 2** (`candidate`, `arena_rejections`, `accepted` = génération + chemin relatif + SHA-256 + mesure contre l'ancre + sa source, `anchor_history`, `anchor_stop_ack`, `commits`, `line_ok_round`, `stop_reason`) ; `_finish_round` (candidat ou rejet) ; `commit_arena(candidat, résumé contre l'accepté, résumé contre l'ancre, ArenaConfig)` (appelle `arena.gate_decision`, écrit `generations/gen_XXXX.pt` avec les résumés des arènes, applique les arrêts) ; `clear_stop` ; `read_train_state` ; migration des états de l'étape 6d. Nouveaux réglages `[selfplay_train]` : `max_arena_rejections = 5`, `anchor_stop_generations = 2`, `anchor_stop_from_generation = 3`.
- **Ordre d'écriture** : fichiers de données (`candidate.pt`, `gen_XXXX.pt`) atomiques, puis `train_state.pt` (point de validation), puis `line_ok.pt` et le journal `rounds.jsonl`. Chaque fichier est rattaché à l'état : `line_ok.pt` porte son numéro de tour (régénéré depuis `train_state.pt` s'il ne correspond pas : à ce moment-là, `train_state.pt` contient exactement l'état du candidat) ; `candidate.pt` et l'accepté sont identifiés par leur SHA-256 ; un `gen_XXXX.pt` ou un `candidate.pt` non référencé est ignoré puis écrasé ; un verdict rejoué ne fait rien (et complète le journal s'il manquait). Les lignes de `rounds.jsonl` ont un `type` (`round`, `arena`, `clear_stop`) et ne sont jamais dupliquées.
- `scripts/train_selfplay.py` : `--commit-arena N --gate <result.json> [--anchor <result.json>]`, `--status`, `--clear-stop "note"` ; code de sortie d'un tour : 0 candidat prêt (ou en attente de données), 1 rejeté, 2 arrêt, 3 interrompu.
- `scripts/selfplay_session.py` : le réseau de self-play vient de `read_train_state` (et non plus du « dernier `gen_*.pt` » du dossier, qui prendrait un orphelin) ; sa boucle provisoire `--with-training` n'a pas d'arène, donc elle garde le réseau de départ jusqu'à l'orchestrateur de 7d.

### Écarts à la conception
- `train_state.pt` est écrit **dès la création** d'un dossier d'entraînement (avant le premier tour), pour que `read_train_state` donne tout de suite le réseau de départ et son SHA-256 (sinon : « pas d'état »). Trouvé par les tests.
- L'écriture atomique des checkpoints force maintenant le fichier sur disque (`fsync`) avant le renommage (déjà le cas dans l'arène).
- Numéro du candidat = **numéro du tour** (graine des ouvertures de l'arène) : les numéros ont des trous quand des tours sont rejetés par les garde-fous.
- Si le candidat perd l'arène contre l'accepté, l'arène ancre n'est pas exigée (le veto ne joue que pour un candidat gagnant) ; si elle manque alors que le candidat gagne (génération ≥ 1), c'est une erreur.
- Le choix « arène de la génération 1 sur les ouvertures fixes de l'ancre » relève de l'orchestrateur (7d) : 7b prend simplement le résumé de l'arène contre l'accepté comme mesure contre l'ancre tant que l'accepté est le réseau de départ.
- La migration écrit `line_ok.pt` puis **renomme `accepted.pt` en `accepted.pt.migre`** (gardé, jamais supprimé ; demande de l'auteur avant le commit).
- Test « rejet par les garde-fous après un rejet en arène » renforcé avant le commit (demande de l'auteur) : il vérifie les deux cas, réseau accepté = réseau de départ PUIS réseau accepté = `gen_0001` ; poids comparés **au bit près** (`torch.equal` sur chaque tenseur) au candidat rejeté en arène et différents de l'accepté ; moments d'AdamW revenus à ceux du candidat.

### Tests (`tests/test_selfplay_candidates.py`, 17 ; `tests/test_selfplay_train.py` adapté ; suite complète : 247 tests en 2 min 02 sur secteur)
- Garde-fous passés → candidat en attente, génération inchangée, shards marqués, `line_ok.pt` au tour du candidat, aucun nouveau tour avant le verdict.
- Première acceptation → `gen_0001.pt` rechargeable, mesure contre l'ancre = arène contre l'accepté (source « biaisée par la sélection ») ; verdict rejoué sans effet ni doublon dans le journal ; mauvais numéro refusé.
- **Rejet en arène → la ligne continue** (poids identiques au candidat rejeté) et **le candidat suivant affronte toujours le même accepté** (SHA-256 de `gen_0001`, pas le candidat rejeté) ; une acceptation remet les rejets à zéro.
- Rejet par les garde-fous après un rejet en arène → retour au candidat rejeté en arène, pas au réseau de départ.
- Veto de l'ancre par `commit_arena` ; arène ancre manquante refusée quand le candidat gagne.
- `decide` reçoit le nombre de générations **acceptées en arène** (0, 0, 0, 0 malgré 4 candidats, puis 1 à 4) ; part humaine 10 % après la 5e acceptation.
- **Arrêt après 5 rejets en arène, les rejets par les garde-fous intercalés ne comptent pas** (4 rejets en arène, 2 par les garde-fous, puis le 5e en arène → arrêt « stagnation ») ; `--clear-stop`.
- Arrêt sur l'ancre, lecture A : sous l'ancre aux générations 2, 3 → pas d'arrêt ; 4 aussi → arrêt (« 3 à 4 ») ; une génération isolée sous l'ancre → pas d'arrêt.
- **Coupures** : après l'écriture de `candidate.pt` (tour non validé → repris → même candidat au bit près qu'un tour d'une traite, une seule ligne de journal) ; entre la validation et `line_ok.pt` (réparé depuis `train_state.pt`, puis un rejet par les garde-fous revient bien au candidat) ; après l'écriture de `gen_0001.pt` (orphelin ignoré, verdict réappliqué, fichier réécrit et référencé) ; entre la validation du verdict et le journal (pas de doublon) ; `gen_0007.pt` et `candidate.pt` orphelins ignorés ; `candidate.pt` modifié refusé (SHA-256).
- Compatibilité : état de l'étape 6d sans génération migré ; avec des générations, refusé.

## Étape 6 — Session complète de self-play seul sur Kaggle (2026-10-22)

Session lancée par l'auteur (code `a50297c`, `MODE = "full"`, `SELFPLAY_ONLY = True`, 300 min, réseau `kaggle_sup`). Sortie téléchargée dans `data/kaggle_out/selfplay_session1/` (hors git : règle `data/*`, vérifié avec `git check-ignore`) ; résumé : `selfplay/sessions/s20261022T133922.json`.
- **Tampon** : 87 shards, **370 840 positions, 3 803 parties**, 18,3 Mo ; chargement en 0,5 s ; `check_selfplay.py --syzygy tools/syzygy` : **aucune erreur**. Parties mises de côté (`crc32(game_uid)`) : 368 (9,7 %), soit **334 393 positions d'entraînement** et 36 447 mises de côté. Résultats : Blancs 1 326, nulles 1 443, Noirs 1 034. Départs : 3 413 du livre (3 384 distincts), 390 de la position initiale ; positions toutes distinctes au demi-coup 30.
- **Débit** : 4 296 simulations/s (1 432 par processus, 3 processus × 64 parties), 793 parties/h, ~77 300 positions/h (1 290 par minute) ; GPU 24 % en moyenne (max 69 %), CPU 76 % ; 219 positions par appel au réseau, réseau 18 % du temps : **le goulot est bien la recherche en Python**. L'essai de débit de 6e donnait ~3 900 simulations/s.
- **Parties** : 111 demi-coups en moyenne (médiane 109), dont 98 joués en self-play ; **97,5 coups cherchés par partie** (370 840 / 3 803). Fins : mat 486, pat 6, 50 coups 6, triple répétition 916, tables 970, abandon 1 419.
- **Abandons** : 367 parties de contrôle ; 154 abandons s'y seraient déclenchés ; **0 faux** (borne haute à 95 % ~2 %, sous la cible de 5 %) : seuil de −0,90 gardé.
- **Répétitions** : 916 nulles par triple répétition (24,1 % des parties, 63,5 % des nulles). Par demi-coup de fin : < 60 : 61 ; 60-99 : 338 ; 100-149 : 418 ; 150-199 : 84 ; ≥ 200 : 15 (médiane 104). Par matériel restant : ≥ 60 : 24 ; 40-59 : 127 ; 20-39 : 401 ; < 20 : 364 (médiane 23).
  - **Conclusion corrigée (auteur)** : les répétitions ne sont PAS surtout des finales : **552 des 916 (60 %) ont lieu avec au moins 20 points de matériel**, soit **14,5 % de toutes les parties** ; 364 seulement (40 %) en finale (< 20 points). Les répétitions précoces (avant le demi-coup 60) restent rares (1,6 % des parties).
  - **Décision de l'auteur** : rien ne change pour l'instant ; on suit le taux de répétition par génération (rapport de session : alerte si la part des parties finies par répétition avec ≥ 20 points dépasse 14,5 % + 5 points) et **on tranche sur la double répétition dans l'arbre après 3 générations acceptées**.
- **Réserve d'arène** : cette session tournait avec le code d'avant l'exclusion de la réserve (`a50297c`) ; **28 parties (0,74 %)** sont parties d'une position de la réserve, dont une seule des 100 ouvertures fixes de l'ancre : négligeable.
- Ce tampon sera l'entrée de la première session de boucle (démarrage à partir d'une sortie de self-play seul, `--expect-session 0`) et la base de l'A/B de la cible de valeur.

## Étape 7d — Orchestrateur de la boucle, budget, rapport (2026-10-22, code pas encore commité)

### Conception validée par l'auteur (avec ses changements)
- **Phase déduite des fichiers durables** à chaque pas (`read_train_state`, `result.json` des arènes, shards du tampon) : arène A si un candidat attend sans résultat contre l'accepté ; arène ancre si A est gagnée et que l'accepté n'est plus le réseau de départ ; verdict ; entraînement si un tour est en cours ou si les positions non encore entraînées atteignent `new_positions_per_generation` ; sinon self-play. Une coupure dans n'importe quelle phase se reprend d'elle-même.
- **`loop/loop_state.json`** ne contient que ce qui appartient à l'orchestrateur : sessions (numéro, début, fin, budget, code git, type d'entrée, raison de fin, rapport), contrôles faits, durées mesurées (estimations), événements des phases, échecs. Jamais le candidat, l'accepté, les compteurs ni l'arrêt (source de vérité : `train_state.pt`).
- **Budget** : un tour ne démarre que s'il reste entraînement + arènes estimés + 15 min ; une arène, sa durée estimée + 15 min ; sinon le reste de la session sert au self-play (données toujours utiles) et le travail attend la session suivante ; 10 min de marge de fin. Tranches de self-play : juste assez pour l'objectif de positions au débit mesuré (au moins 10 min ; plus de nouvelles parties 12 min avant la fin d'une tranche). Échéances dures transmises aux sous-processus (`arena.py --max-minutes`, **`train_selfplay.py --max-minutes`** ajouté : arrêt propre, reprise au pas près).
- **Estimations** : moyenne des 3 dernières mesures dans `loop_state.json`. **Valeurs de départ tirées de la session Kaggle ci-dessus** : 1 290 positions/min ; arène = 97,5 coups cherchés par partie × 200 simulations / 4 296 simulations/s × 1,10 (deux lots de demi-taille, fin des longues parties) = **0,083 min par partie, 17,6 min pour 200 parties** (+ 1 min de démarrage) ; entraînement 5 min + 0,006 min par pas. Les contrôles d'arène alimentent aussi l'estimation des arènes (manque trouvé par les tests).
- **Génération 1** : tant que l'accepté est le réseau de départ, l'arène contre l'accepté se joue sur les ouvertures fixes de l'ancre (`--kind anchor`), sans arène ancre séparée.
- **Contrôles d'arène à la première session de boucle** (auteur) : **(a) non bloquant** (sur GPU l'inférence par lots n'est pas garantie déterministe, les parties peuvent diverger et (a) devient un vrai match : hors de 40-60 % environ 3 fois sur 10 par hasard sur 20 parties ; et il ne détecte pas une inversion des réseaux) : avertissement dans le rapport avec le score et la symétrie des paires ; **(b) bloquant** : 200 contre 50 simulations, 20 ouvertures, seuil 70 %.
- **Protection contre une mauvaise entrée** (auteur) : `--expect-session N` obligatoire = numéro de la dernière session de boucle de l'entrée (0 pour une sortie de self-play seul) ; sinon refus avec les numéros trouvé et attendu. Le rapport donne en tête la valeur de `EXPECTED_SESSION` pour la suivante.
- **Démarrage à partir d'une sortie de self-play seul** (auteur) : sortie avec `selfplay/` seulement → message « Démarrage de la boucle à partir d'une sortie de self-play seul, n positions dans le tampon », `train_state.pt` créé avec `kaggle_sup` comme accepté, `loop_state.json` à la session 1 ; refus sous 250 000 positions d'entraînement (nombre trouvé donné) ; refus d'une sortie incohérente (tampon + état de boucle sans état d'entraînement, tampon + `sp_train/` sans état de boucle, etc.). Le SHA-256 de l'ancre est enregistré : une session suivante refuse un autre réseau de départ.
- **Nouvel essai** (auteur) : une phase en échec (code inattendu d'un sous-processus) est relancée une fois (elle se reprend d'elle-même) ; le premier échec est journalisé avec la fin du message d'erreur ; arrêt au deuxième échec de la même phase. Les arrêts décisionnels (G4, stagnation, ancre, contrôle (b)) sont immédiats. Tout arrêt : rapport puis fin de session.
- **Rapport** `loop/reports/session_XXXX.md` (+ `.json`) : `EXPECTED_SESSION` en tête, « Actions pour toi » (arrêt et décision à prendre, erreur, contrôle (b) échoué, avertissement du contrôle (a), mesure v2 recommandée — Elo contre l'ancre lissé sur 3 générations > +50 depuis la dernière mesure, ou ≥ 10 générations — d'après les `docs/stockfish/v2/sp_gen*.json` du dépôt, faux abandons > 5 %, répétitions précoces > 5 %, **répétitions avec ≥ 20 points de matériel > 14,5 % + 5 points**, alertes des garde-fous, biais de la génération 2, baisse de débit > 15 %, entrée de la session suivante), notes de démarrage et contrôles, candidats (pas, garde-fous, arène A avec Elo ± intervalle et probabilité d'être meilleur, ancre, décision), état final, self-play (débit, fins, répétitions, faux abandons), chronologie (avec la fin du message d'erreur des échecs). `loop/history.csv` (une ligne par verdict, reconstruite à chaque session depuis `rounds.jsonl`) et `loop/progress.png` (Elo contre l'ancre par génération avec intervalles et mesures v2 ; top-1 Lichess et erreur de valeur des parties mises de côté par candidat).

### Code
- `src/chesszero/loop.py` : `LoopConfig` (section `[loop]`), état de l'orchestrateur, `classify_output`, `check_expected_session`, `next_phase`, `selfplay_chunk`, estimations, `SubprocessExecutor` (lance `selfplay.py`, `train_selfplay.py`, `arena.py`, garde la fin de leur sortie), `Orchestrator` (démarrage, contrôles, phases, nouvel essai, fin), `actions`, `write_report`, `history_rows`, `plot_progress`, `measured_v2_generations`.
- `scripts/alphazero_session.py` (`--work`, `--resume-from`, `--expect-session`, `--minutes` ≤ 660, `--init`, `--human`, `--smoke` ; code de sortie 0 fin de budget, 2 arrêt, 4 refus de démarrer) ; `configs/smoke.toml` (essai local, réseau tiny, tout réduit) ; section `[loop]` de `kaggle.toml`.
- `selfplay_train.py` : `run_round(deadline=)`, `read_train_state` donne aussi `trained_shards` et `init_sha256`, `export_trained` (A/B) ; `train_selfplay.py --max-minutes`, `--export-trained`.

### Écarts à la conception
- **Pas de cache Lichess** : charger les 23,6 M positions prend 2,7 s (fichiers `.npz` non compressés) ; le cache n'apportait rien.
- Les contrôles d'arène alimentent l'estimation des arènes (sans cela, la première vraie arène était estimée avec la valeur de départ ; trouvé par les tests).
- SHA-256 de l'ancre dans `train_state.pt` (`init_sha256`) pour refuser un autre réseau de départ.
- Seuil du contrôle (b) à 50 % dans `configs/smoke.toml` seulement (6 parties d'un réseau tiny, 16 contre 1 simulation) ; 70 % sur 40 parties sur Kaggle.
- Pour l'A/B : `--export-trained` (réseau du tour gardé même s'il est rejeté par les garde-fous).

### Tests (`tests/test_loop.py`, 31)
Faux exécuteur (phases en mémoire, horloge simulée, mêmes codes de sortie que les scripts) ; tampon de départ produit par le **vrai `scripts/selfplay.py`** (format exact d'une session de self-play seul).
- Fonctions pures : phase suivante (table de cas), tranches de self-play et budget, messages de la session attendue, « Actions » (mesure v2, faux abandons, répétitions, arrêts, biais de la génération 2, débit), **alerte de répétitions avec ≥ 20 points** (la session de référence n'alerte pas ; 19 % non ; 20 % oui), mesures v2 lues dans le dépôt.
- **Démarrage** à partir d'une sortie de self-play seul (message, état d'entraînement avec l'ancre, session 1, statistiques de l'étape 6 recopiées, `--expect-session 1` refusé) ; **refus sous le tampon minimal** (nombre donné) ; entrées incohérentes refusées ; **sortie plus ancienne mais valide refusée** (« trouvé la session n°1, attendu la n°2 »).
- Session complète : contrôles puis génération 1 sur les ouvertures fixes de l'ancre, arènes ancre ensuite, self-play toujours avec le réseau accepté, rapport (`EXPECTED_SESSION`, Actions, biais de la génération 2), `history.csv`, `progress.png`, estimations mesurées.
- Contrôle (a) hors de l'intervalle : avertissement, la session continue ; contrôle (b) raté : arrêt immédiat, aucun entraînement.
- **Nouvel essai** : un échec de self-play rejoué, session normale, échec journalisé avec la fin du message ; deux échecs → arrêt « erreur » ; arrêt décisionnel (G4) immédiat sans nouvel essai.
- Budget : arène reportée à la session suivante, reste du temps en self-play, la session suivante commence par cette arène.
- **Coupure simulée à chacun des 13 points de l'orchestrateur** (après le démarrage, avant / après chaque phase, entre le verdict et l'enregistrement, avant le rapport) puis reprise : un verdict par candidat, générations cohérentes, aucune arène rejouée, tampon intact, deux sessions enregistrées.
- Dossier d'arène d'un autre candidat mis de côté (`.stale`) puis rejoué ; dossiers des candidats déjà décidés ignorés (fichiers orphelins au sens de l'orchestrateur).
- **Tampon de départ de la première session de boucle** (proportions réelles) : entraînement immédiat ; 1er tour plafonné à 435 pas (un passage moyen par position, au lieu de 1 742 pour 4 passages) ; puis self-play de 100 000 positions (~77,5 min) ; 2e tour 469 pas ; 1er tour + arène de la génération 1 estimés à ~25 min.
- `tests/test_selfplay_candidates.py` : export du réseau du tour même rejeté.

### Tampon de départ : déroulé prévu de la première session de boucle (estimations)
Entrée : les 370 840 positions ci-dessus (334 393 d'entraînement), aucune encore utilisée, objectif 100 000 par génération.
- Démarrage (~1 min) : copie du tampon (18 Mo), `train_state.pt` avec `kaggle_sup` accepté ; contrôles (a) 20 parties et (b) 40 parties, ~7 min.
- **Tour 1 tout de suite** (tout le tampon est « nouveau ») : **435 pas** (plafond d'un passage moyen par position ; 4 passages demanderaient 1 742 pas), ~8 min ; puis **arène de la génération 1** (200 parties sur les ouvertures fixes de l'ancre, ~18 min) et verdict : vers T + 35 min.
- Tous les shards sont alors marqués comme utilisés (même si l'arène est perdue : la ligne continue) : **self-play de 100 000 positions** avec le réseau accepté (~78 min), puis tour 2 (90 000 positions d'entraînement nouvelles sur ~424 000 : 469 pas, 4 passages), arène A et arène ancre (si l'accepté est la génération 1), verdict : vers T + 155 min.
- Les positions de départ restent dans la fenêtre de 500 000 : elles sont revues dans les tours suivants (~0,85 passage au tour 2) jusqu'à ce que la fenêtre glisse (après ~2 générations) ; au total ~2 à 3 passages chacune.
- Avec une session courte (~220 min) : après le 2e verdict, plus assez de temps pour 100 000 positions + un tour : self-play jusqu'à la fin (~50 min, ~65 000 positions), rapport. La session suivante commence par ~35 min de self-play puis le tour 3.
- Si le tour 1 est rejeté par les garde-fous : ses shards ne sont pas marqués, le tour suivant reprend les mêmes 370 840 positions.

### Décomposition du temps d'entraînement (demande de l'auteur)
- **Précision sur l'hypothèse de départ** : les `.npz` Lichess ne sont PAS projetés en mémoire : `load_split` (`np.load` sans `mmap_mode`) lit chaque tableau en entier dans la RAM au chargement (1,8 Go pour `lichess_24m`), puis les lots sont tirés en RAM. Sur Kaggle, le risque est donc la **lecture séquentielle initiale depuis `/kaggle/input`**, pas l'accès aléatoire ; les 2,7 s mesurées en local (SSD, fichiers peut-être déjà en cache) ne disent rien de Kaggle.
- Chaque tour enregistre maintenant (`rounds.jsonl`, champ `timing`, cumulé sur les reprises) : **chargement** des données Lichess de ce lancement (+ masques de validation), **lots Lichess** (lecture des positions, coups, résultats), **tampon** (chargement des shards au début du tour + lecture des positions et cibles), **masques Lichess** (attente du pool), **préparation** (décodage en tenseurs), **calcul** (copie vers le device, passe avant/arrière, optimiseur, synchronisé sur CUDA), **évaluation** (Lichess + parties mises de côté), **autre**, et le temps total réel. Le rapport de session donne ces parts pour chaque tour ; **« Actions pour toi » signale un tour dont la lecture (chargement + lots Lichess + tampon) dépasse 30 %** (`reading_alert`). Tests : décomposition enregistrée et cohérente après une reprise (`tests/test_selfplay_train.py`), seuil de l'alerte (`tests/test_loop.py`).

### Essai complet local `--smoke` (2026-10-22, sur secteur, PC sans autre charge)
Réseau `tiny_sup`, configuration `smoke` (16 simulations, 3 ouvertures par arène, lots de 64, objectif 1 500 positions par génération), CPU, sans tables.
- **s0 = sortie de self-play seul** produite par le vrai `selfplay_session.py full` (3 min) : 363 parties, **24 413 positions** (21 565 d'entraînement), soit ~16 fois l'objectif par génération (la vraie : 370 840 pour 100 000). Défaut mineur de cet ancien script vu au premier essai : sa tranche minimale par défaut (30 min) empêche toute partie dans une session de 3 min, puis son rapport plante sur des statistiques vides (relancé avec `--min-chunk-minutes 0.5` ; sans effet sur la boucle).
- **Session 1** (`--resume-from s0 --expect-session 0`, 8 min, 21 h 28 → 21 h 36, code 0) : message de démarrage ; contrôle (a) 4 parties 50 %, 2/2 paires symétriques ; contrôle (b) (16 contre 1 simulation) 83 % ; tours 0 et 1 rejetés par les garde-fous (G1 / G2 : taux ÷ 2 deux fois) ; tour 2 candidat → arène de la génération 1 sur les ouvertures fixes de l'ancre, 50 % → rejeté (la ligne continue) ; self-play ; tour 3 → 41,7 % → rejeté ; self-play ; tour 4 → 58,3 % → **accepté (génération 1)** ; self-play final jusqu'à la fin du budget. Rapport, `history.csv`, `progress.png` produits ; « Actions » : mesure v2 recommandée (+58 lissé), faux abandons (réseau tiny), alertes G3, une alerte de lecture (tour 4 : 30 % dont 27 % de chargement, car ce tour ne durait que 0,3 min).
- **Session 2** (`--resume-from s1 --expect-session 1`, 6 min, 21 h 36 → 21 h 42, code 0) : entrée reconnue comme sortie de boucle, contrôles non rejoués ; tour 5 tout de suite (données du self-play final de la session 1) rejeté par les garde-fous ; tour 6 candidat → arène contre `gen_0001` 50 % → rejeté ; self-play ; tour 7 → 50 % → rejeté (2 rejets en arène d'affilée) ; self-play final. Le rapport rappelle le biais de la génération 2.
- Non exercé par l'essai (couvert par les tests) : l'arène ancre (aucun candidat n'a gagné contre `gen_0001`), une coupure au milieu d'une phase (les deux sessions se sont terminées sur un self-play final), le nouvel essai après un échec.
- Les chiffres (débit, faux abandons, Elo) n'ont aucun sens avec un réseau tiny à 16 simulations : seule la mécanique compte.

### A/B de la cible de valeur : préparé, pas lancé (`scripts/ab_value_target.py`)
`ab_value_target.py --buffer data\kaggle_out\selfplay_session1\selfplay --out data\ab_value` : (1) étiquetage Stockfish de 20 000 positions des parties mises de côté (sur 36 447, 368 parties) + 20 000 de la validation Lichess, 200 ms par position, 8 processus ; (2) un tour `result` puis un tour `blend` depuis `kaggle_sup` (mêmes lots : même graine, même tour 0, même tampon ; 435 pas de 1 024 positions chacun ; `--export-trained`) ; (3) `compare`. Reprise : une étape faite est sautée, un tour coupé reprend au pas près. Avertissement si le PC n'est pas sur secteur en « meilleures performances ».
- **Durée estimée : ~1 h 35 à 2 h** sur secteur : étiquetage ~17 min (40 000 × 0,2 s / 8) ; chaque tour ~35 à 40 min (mesure : **4,4 s par pas** de 1 024 positions sur l'i9 à 14 threads, un peu pessimiste car mesurée pendant une autre tâche ; 435 pas ≈ 32 min, + évaluations Lichess et parties mises de côté ~3 min) ; comparaison ~2 min.

## Bilan de la chaîne de nuit du 2026-10-22 (relu le 2026-10-23)

Chaîne lancée dans un processus détaché (`data\nuit\chaine_nuit.bat`, PID 22892) : (1) A/B de la cible de valeur depuis le dépôt ; (2) seconde mesure v2 de `kaggle_sup` (tag `kaggle_sup_r2`, `--like kaggle_sup`) depuis la copie figée `..\Projet_chess_nuit` au commit `c995d12`.

### Déroulement
- 21 h 55 → 0 h 58 : A/B, code 0 (182,7 min au lieu de ~1 h 35 à 2 h). 0 h 58 → 6 h 58 : mesure v2, code 0, 640 parties (**360,4 min au lieu de ~142**). Aucune erreur dans `1_ab_value.log` ni `2_stockfish_v2.log`.

### Alimentation et vitesse : mesure v2 NON VALIDE (processeur bridé toute la nuit)
- **Secteur toute la nuit** (journal Système, Kernel-Power 105) : batterie de 21 h 14 (`AcOnline=false`) à **21 h 54 min 28 s** (`AcOnline=true`), aucun changement ensuite ; batterie 71 % au départ de la chaîne, 100 % au début de la mesure v2 ; mode « meilleures performances » relevé au départ des deux étapes. Aucun événement de veille ni de bridage thermique dans le journal Système.
- **Mais le processeur a tourné ~3 fois moins vite, du premier au dernier coup** : 2,48 s par coup de l'IA en médiane (référence `v2/kaggle_sup` : 0,80 s ; aucune partie sous 1,25 s ; médiane par tranche de 80 parties constante entre 2,42 et 2,55) ; l'entraînement de l'A/B à ~102 positions/s (10 s par pas au lieu de ~4,4 s).
- Ce matin, même machine, sur secteur : `match_stockfish.py bench --bench-games 8` = **0,76 s par coup**, lancé normalement **et** lancé détaché par WMI comme la chaîne (`Win32_Process.Create`) : le mode de lancement n'est pas en cause. **Cause non identifiée** ; hypothèse principale : bridage du portable quand l'utilisateur est absent (capot fermé ou écran éteint ; Intel DTT `ipf_helper` est actif), à tester.
- **Effet sur la mesure** : Stockfish joue au temps (300 ms). Bridé (profondeur sondée à 100 ms au lieu de 300 ms, ~60 positions des parties de la nuit, `UCI_Elo` 2500, 1 thread) : profondeur médiane **12 au lieu de 14** (10e centile 10 au lieu de 12). Or `UCI_Elo` choisit son coup à la profondeur 1 + niveau : 10 pour 2200-2250, 12 pour 2350, 13 pour 2450, **14 pour 2500** (déjà juste atteinte à pleine vitesse). Les niveaux ≥ 2350 ont donc joué plus faible que dans la référence. **Décision (règle des mesures) : ne pas combiner `kaggle_sup_r2` avec la référence.**
- Pour information seulement : 2428 ± 36 (2392 à 2463) ; `compare` : +1 ± 51 (scores par niveau 100 / 94 / 80 / 68 / 76 / 67 / 62 / 47 % de 1320 à 2500). Fichiers gardés hors du dépôt dans `data\nuit\` (`kaggle_sup_r2.json`, `.pgn`, `2_stockfish_v2.log`), **pas dans `docs/stockfish/v2/`**. La référence reste `v2/kaggle_sup` seule (2427 ± 37, écart minimal significatif ~55 Elo).

### A/B de la cible de valeur (`data\ab_value\compare.txt`)
Mêmes lots, 435 pas chacun depuis `kaggle_sup` ; 40 000 positions fixes (20 000 des parties mises de côté, 368 parties ; 20 000 de la validation Lichess) notées par Stockfish plein à 200 ms (étiquetage fait processeur bridé : profondeur moindre, mais **mêmes étiquettes pour A et B**, la comparaison reste équitable). Erreur quadratique, intervalle à 95 % par bootstrap apparié sur les parties :

| Mesure | `kaggle_sup` (départ) | `result` | `blend` | blend − result |
|---|---|---|---|---|
| contre Stockfish, toutes positions | 0,1937 | **0,1801** | 0,1906 | **+0,0104 (+0,0091 à +0,0116)** |
| contre Stockfish, self-play | 0,2189 | **0,1986** | 0,2095 | +0,0108 (+0,0085 à +0,0132) |
| contre Stockfish, Lichess | 0,1685 | **0,1616** | 0,1716 | +0,0100 (+0,0092 à +0,0109) |
| contre le résultat réel, self-play mis de côté | 0,3820 | 0,3745 | 0,3730 | −0,0015 (−0,0056 à +0,0028) |

- Les deux tours améliorent le réseau de départ sur les deux mesures (`result` − départ : −0,0136 contre Stockfish, −0,0075 contre les résultats ; `blend` − départ : −0,0031 et −0,0090, tous significatifs).
- **Contre Stockfish, `result` est significativement meilleur** ; contre les résultats réels, **égalité** (intervalle centré sur 0). Verdict du script : mesures de sens opposés → garder `result`.
- Explication probable : le réseau de départ est trop prudent (pente de la régression Stockfish ~ valeur : 1,22 ; |valeur| moyenne 0,35 contre 0,54 pour Stockfish). `result` le rend plus tranché (pente 1,16, |v| 0,38) ; `blend` le tire vers la Q racine, calculée par ce même réseau prudent (pente 1,26, |v| 0,34), et renforce donc son biais. La réduction de variance attendue du mélange ne se voit pas sur les résultats réels.
- Autres chiffres : valeur Lichess `result` 0,7355 (départ 0,7267, +0,009, sous l'alerte G3 de +0,05), `blend` 0,7267 ; politique Lichess 1,4239 / 1,4230 ; top-1 53,3 % pour les deux.
- **Recommandation : garder `value_target = "result"`** (aucun changement de code). À reconsidérer seulement si la Q racine vient un jour d'un réseau nettement plus fort que celui qui apprend.

### Script
- `ab_value_target.py` : trois options d'essai réduit transmises aux deux tours (`--steps`, `--batch-size`, `--lichess-val`), utilisées pour vérifier la chaîne avant la nuit.

### Mémoire sur Kaggle (mesures sur le PC, CPU)
- **Tour d'entraînement** (`train_selfplay.py`, 30 pas, tampon de 370 840 positions, `lichess_24m`) : processus principal **~3,4 à 3,6 Go stables** (23,6 M positions Lichess = 1,8 Go, + torch, réseau, validation Lichess de 50 000 positions, tampon) ; pics à 6,3 Go pendant les pas, dus aux activations du lot de 1 024 calculées **sur CPU** (sur Kaggle elles sont sur le GPU) ; 3 processus de masques à ~0,64 Go chacun (Windows = spawn, chacun importe torch ; sous Linux = fork, pages partagées). Pic de chargement estimé à ~+1,6 Go (`load_split` concatène shard par shard). Tampon de 500 000 positions : ~0,1 Go.
- **Self-play** (1 processus × 64 parties, 4 min) : ~1,1 à 1,2 Go privés, en lente hausse (arbres) ; sur GPU, ajouter le contexte CUDA côté hôte (~1 à 2 Go par processus, estimation non mesurée).
- **Bilan Kaggle** (~31 Go) : phase d'entraînement ≈ 6 à 8,5 Go (principal avec CUDA ~5 à 6,5 Go au pic + masques ≤ 2 Go) ; phase de self-play ou d'arène ≈ 3 × 3 à 3,5 Go ≈ 9 à 10,5 Go ; l'orchestrateur arrête le self-play pendant l'entraînement, donc pas de cumul. Même cumulés : ~19 Go. **Marge confortable : rien à changer.** À confirmer par une mesure de la RAM sur la première session de boucle.

## Décisions du 2026-10-23, indicateur d'accord, surveillance de la vitesse, test du capot, étape 7e

### Décisions de l'auteur
- **Répétitions** : les parties lues montrent un réseau qui répète dans des positions qu'il juge égales (Q entre −0,4 et +0,2), pas un bug. **On n'ajoute pas la règle de double répétition** ; on garde seulement l'alerte de suivi (répétitions avec ≥ 20 points de matériel > 14,5 % + 5 points, répétitions précoces > 5 %).
- **Cible de valeur** : `value_target = "result"`, **question close** ; on n'y reviendra que si l'erreur de valeur sur les parties mises de côté stagne pendant plusieurs générations.
- **Mesure v2 de nuit écartée** ; la référence reste `v2/kaggle_sup` seule (2427 ± 37). **Mémoire Kaggle** : ni sous-ensemble ni projection en mémoire.

### Analyse du tampon de la génération 0 (`scripts/analyse_generation.py`, résultats dans `docs/analyse_gen0/`)
- Les shards permettent de rejouer les parties coup par coup (coups UCI depuis la position initiale, ouverture du livre comprise, résultat, raison de fin, Q racine de chaque position cherchée) : 40 505 positions rejouées, 0 incohérence. Les coups forcés ne sont pas enregistrés ; le dernier Q enregistré peut précéder la fin de 1 demi-coup (arbitrage).
- PGN commentés (`repetitions_40pts.pgn`, `mats_les_plus_courts.pgn`, `abandons.pgn`, `README.md`), **commités avec le reste** : répétitions avec ≥ 40 points de matériel : 151 sur 916 (53 avant le 60e demi-coup, 98 après), 5 + 5 tirées (graine 0) ; 5 mats les plus courts (35 à 41 demi-coups) ; 5 abandons (graine 0). Pas de pic de fins de répétition à 40 demi-coups (5 parties sur 916) : les trois parties du tirage qui finissent à 40 ne sont pas la règle.
- **Accord entre la politique brute de `kaggle_sup` et la cible MCTS**, 20 000 positions d'entraînement (graine 0, parties mises de côté exclues), `docs/analyse_gen0/accord_politique.json` : **tout 59,8 % (KL 0,535)** ; ouverture (< 30 demi-coups) 66,2 % (0,492) ; milieu 55,7 % (0,582) ; finale (≥ 30 demi-coups et < 20 points de matériel) 66,9 % (0,428). Le réseau est moins tranché que la cible (probabilité de son coup préféré 0,53, part de visites du coup le plus visité 0,62). Biais : visites issues d'une recherche avec bruit de Dirichlet (ε 0,25) ; à comparer d'une génération à l'autre, pas à une valeur absolue.
- Scripts : `analyse_generation.py check|pgn|agreement` (la commande `pgn` avec `--name "génération 0" --date 2026.10.22 --player "kaggle_sup (200 sim.)"` redonne les fichiers PGN à l'identique, vérifié par comparaison d'octets ; `agreement` redonne les mêmes chiffres) ; tests `tests/test_analyse_generation.py`.

### Indicateur d'accord dans le rapport de chaque session (`src/chesszero/agreement.py`)
- Calculé **en fin de `scripts/selfplay.py`** (donc à chaque tranche de self-play) : échantillon de 5 000 positions à graine fixe parmi les parties de la tranche, accord et KL **par phase** (ouverture / milieu / finale), avec **deux réseaux sur le même échantillon : le réseau qui a joué les parties ET `kaggle_sup`** (option `--reference`, passée par l'orchestrateur). L'écart entre les deux montre ce que le réseau a appris depuis le départ. Écrit dans `sessions/<nom>.json` (clé `agreement`) ; une erreur ne fait jamais échouer le self-play (`{"error": ...}`).
- Le rapport de session fusionne les tranches (moyenne pondérée par le nombre de positions) et donne : les deux réseaux, l'écart en points, la référence de la génération 0, **et le rappel du biais du bruit de Dirichlet**. Option `--agreement-positions 0` pour le désactiver. Tests : `tests/test_agreement.py` (calcul vectorisé contre une boucle position par position, valeurs exactes avec des réseaux factices, fusion pondérée, texte, bout en bout avec le vrai `selfplay.py`).

### Surveillance de la vitesse pendant les mesures (`src/chesszero/speedwatch.py`, `match_stockfish.py`)
- `run` joue d'abord un **bench de départ** (une partie par processus, non enregistrées) = vitesse de référence de la mesure ; avertissement si elle s'écarte de plus de 25 % de la vitesse nominale (0,80 s par coup, mesure v2 de référence). Ensuite la **moyenne glissante sur les 20 dernières parties** du temps par coup de l'IA est suivie : affichée sur chaque ligne (`moy.20 0.81 s (+1%)`), enregistrée dans chaque partie (`ai_rolling_s_per_move`) et dans le fichier de résultats (clé `speed` : référence, benches, série, épisodes d'alerte, pire écart). **Bandeau d'avertissement** quand elle sort de ±25 % de la référence, message de retour à la normale ensuite.
- **Résumé final** à quatre états : stable / perturbée brièvement (alerte sur ≤ 5 % des parties et terminée) / **NON STABLE** / **NON COMPARABLE** (stable mais vitesse de départ hors ±25 % du nominal : le cas de la nuit du 2026-10-22 serait ressorti ainsi, avec le bandeau dès le bench de départ). Reprise : la référence du premier lancement est conservée.
- **Calibrage sur la mesure de référence** (rejeu de `v2/kaggle_sup`) : elle sort deux fois de la tolérance (parties 85-94, +26 %, et 522-541, +44 % : quatre parties simultanées à 1,9-2,3 s par coup), 28 parties sur 640 (4,4 %) : « perturbée brièvement ». Cela ne la remet pas en cause (épisodes courts), mais c'est un fait nouveau à connaître. Essai réel de 3 min sur le PC : bench de départ 0,72 s par coup, moyenne glissante 0,78 s, aucune alerte, données bien écrites dans le JSON. `--no-speed-watch` désactive. Tests : `tests/test_speedwatch.py`.

### Test du capot (2026-10-23, 12:34 → 12:39, sur secteur, auteur présent)
- Action du capot sur secteur : « Ne rien faire » (sur batterie : veille) ; écran jamais éteint sur secteur. Bench de 16 parties (4 à la fois) avec un enregistreur de fréquence processeur (WMI, toutes les 2 s) ; bips : fermer à 12:35:58, rouvrir à 12:38:08. Résultats : fréquence ~2 380 à 2 400 MHz et performance 116 % avant **et pendant** la période ; temps par coup de l'IA 0,65 à 0,84 s (0,78 s en moyenne sur 16 parties, ce matin 0,76 s). **Aucun ralentissement : le capot fermé 2 min ne reproduit pas la nuit** (confirmation de l'auteur attendue que le capot était bien fermé pendant ce créneau).
- La cause du ralentissement de la nuit reste **inconnue**. Pistes non testées : écran verrouillé, absence prolongée (veille d'écran, processus lancés détachés mis en arrière-plan), température sur plusieurs heures, passage batterie → secteur à 21 h 54 (le PC était sur batterie de 21 h 14 à 21 h 54). Outil retenu : la surveillance de la vitesse ci-dessus, qui attrape le problème au départ ou en cours de route.

### Étape 7e — Notebook de la boucle, mode check, mémoire
- **Notebook** `notebooks/kaggle_selfplay.ipynb` : modes `check` (défaut, sans GPU, **rien d'écrit**), `loop`, `bench`, `full`. `LOOP_MINUTES = 220` pour la première session (660 ensuite), `EXPECTED_SESSION`. **`COMMIT` figé** : hash complet obligatoire (la cellule 0 refuse `None`, `"main"`, un hash court) ; la cellule 1 clone le dépôt, se place sur ce commit (`git checkout --detach`) et vérifie `HEAD == COMMIT` ; le code de chaque session est donc reproductible. Un test vérifie que le `COMMIT` du notebook existe dans le dépôt et contient les fichiers dont le notebook a besoin.
- **Découverte des entrées par le contenu** (`src/chesszero/kaggle_inputs.py`) : réseau de départ par SHA-256, données Lichess (dossier avec `train_00000.npz` + `val_*.npz`), sortie précédente (dossier avec `selfplay/` aux shards complets, type et numéro de session lus dans les fichiers). **Exactement une sortie précédente** : zéro → refus (« Aucune sortie précédente… Add Input → Your Work »), deux → refus avec les deux listées ; `EXPECTED_SESSION` incohérent → refus avec la valeur à mettre.
- **Mode check** : `alphazero_session.py --check-inputs` (+ `--expect-init-sha256`, refus si le réseau de départ diffère) : toutes les vérifications de démarrage (`Orchestrator.validate_input`, utilisée aussi par `prepare`), réseau chargé, Lichess listé, RAM disponible ; code 0 = la session démarrerait, 4 = refus ; **ne crée aucun fichier** (le dossier de travail n'existe même pas ; la cellule du notebook vérifie que `/kaggle/working` est resté vide).
- **Mémoire** (`src/chesszero/memwatch.py`) : la RAM disponible du système (tous les processus) est relevée toutes les 30 s dans un thread de l'orchestrateur ; minimum global et par phase, relevés sous 4 Go ; section « Mémoire » du rapport, `ram` dans `loop_state.json`, action **MÉMOIRE BASSE** sous 4 Go (`[loop] ram_interval_seconds`, `ram_alert_gb`).
- Tests : `tests/test_loop_inputs.py` (arborescence `/kaggle/input` simulée avec les deux dispositions de montage, zéro sortie, deux sorties, mauvaise session, réseau au mauvais SHA-256, Lichess absent ou sans validation, shard incomplet ; mode check sans écriture et ses refus ; ligne de commande ; mémoire ; notebook).
