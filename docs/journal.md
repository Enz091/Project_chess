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
