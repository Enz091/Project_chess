# Projet_chess — IA d'échecs façon AlphaZero

> **RÈGLE — Python sur cette machine** : `python` n'est pas dans le PATH (Windows ouvre le
> Microsoft Store à la place). Toujours utiliser `.venv\Scripts\python.exe` pour lancer des
> scripts, les tests (`-m pytest`) ou pip (`-m pip`). Ne jamais appeler `python`, `pip` ou
> `pytest` directement. Dans un shell bash, écrire `.venv/Scripts/python.exe`, et éviter de
> passer des chemins à antislashs dans du code Python inline (ils sont mal interprétés).

IA d'échecs en Python (PyTorch + python-chess) : un réseau de neurones à deux têtes
(**politique** = probabilité de chaque coup, **valeur** = qui gagne) guide une recherche
**MCTS**. Le réseau est d'abord **pré-entraîné en supervisé** sur des parties Lichess, puis
amélioré par **self-play**.

L'auteur veut comprendre l'architecture globale : expliquer chaque choix avant de l'appliquer,
avancer étape par étape, et **mettre à jour ce fichier (section Avancement) à la fin de chaque étape**.

## Environnement

- **Local (développement / tests)** : Windows 11, Intel Iris Xe, pas de GPU NVIDIA → CPU.
  Python 3.12 (le projet exige ≥ 3.11 pour `tomllib`), venv dans `.venv/`, PyTorch **version CPU**.
- **Entraînements réels** : Kaggle (GPU), le code arrive par GitHub (`git clone`).
- Git : branche `main`, auteur `345045991+Enz091@users.noreply.github.com` (config locale au dépôt).
  Dépôt GitHub : `https://github.com/Enz091/Project_chess.git` (remote `origin`, `main` poussée).

### Installation locale
```powershell
py -3.12 -m venv .venv    # lanceur « py » (présent ici) ; python n'est pas dans le PATH
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
```

### Installation sur Kaggle (notebook)
```bash
git clone https://github.com/<compte>/<depot>.git && cd <depot>
pip install -e .        # NE PAS réinstaller torch : garder le torch CUDA de Kaggle
```

## Choix techniques

| Choix | Raison |
|---|---|
| torch absent de `pyproject.toml` | Torch CPU en local et torch CUDA sur Kaggle : chaque environnement installe le sien. |
| Package dans `src/chesszero/`, installé avec `pip install -e .` | Les mêmes imports marchent partout, et les modifications du code sont prises en compte sans réinstaller. |
| `device.get_device()` : CUDA > XPU > CPU | Pas de `"cuda"` en dur. On peut forcer un choix avec `--device` ou la variable `CHESSZERO_DEVICE`. |
| Checkpoints complets | Poids, optimiseur, scheduler, step/epoch, états RNG, config et métriques : une reprise est identique à un run sans coupure (vérifié par un test). |
| Écriture atomique (`.tmp` puis `os.replace`) | Une coupure Kaggle pendant la sauvegarde ne corrompt pas `last.pt`. |
| `last.pt`, `best.pt`, rotation des `keep_last` derniers `ckpt_XXXXXXXX.pt` | Reprise automatique depuis `last.pt` sans remplir le disque. |
| Chargement avec `map_location="cpu"` | Un checkpoint GPU (Kaggle) se recharge sur le PC, et inversement. |
| `weights_only=False` au chargement | Nécessaire pour les états RNG Python/NumPy. Ne charger que nos propres checkpoints. |
| `data/` et `checkpoints/` hors de git | Trop volumineux. Sur Kaggle, ils passent par les Datasets et le dossier de sortie. |
| Scripts dans `scripts/`, logique dans le package | Le notebook Kaggle reste minimal : il clone le dépôt puis lance un script. |

## Arborescence

```
src/chesszero/   device.py, checkpoint.py, encoding.py, model.py, data.py, train.py, mcts.py   (+ selfplay à venir)
scripts/         points d'entrée (smoke_resume.py = démo reprise, show_encoding.py = démo encodage, bench_inference.py = vitesse du réseau, prepare_data.py = extraction Lichess (un ou plusieurs mois), make_kaggle_zip.py = archive des shards pour un Dataset Kaggle, summarize_run.py = débit/GPU/attente d'un entraînement + verdict, train.py = pré-entraînement, plot_curves.py = courbes PNG, show_policy.py = meilleurs coups d'un checkpoint, show_search.py = ce que voit le MCTS sur une position, bench_mcts.py = simulations/s selon la taille de lot, match.py = MCTS contre politique seule)
configs/         tiny.toml (4 blocs × 64, CPU) et kaggle.toml (10 blocs × 128), sections [model], [train], [mcts] ; data.toml = filtres Lichess
tests/           pytest
docs/            journal.md = détails, mesures et tests de chaque étape ; img/ = courbes d'entraînement, match_*.pgn = parties des matchs
notebooks/       kaggle_train.ipynb = essai de débit + entraînement sur Kaggle (clone + lance des scripts)
data/            PGN Lichess et données prétraitées (ignoré par git)
checkpoints/     sauvegardes d'entraînement (ignoré par git)
```

## Feuille de route

0. **Mise en place** : venv, PyTorch CPU, python-chess, git, arborescence, détection du device, checkpoints avec reprise.
1. **Encodage** : plateau vers tenseur (plans de pièces, roques, en passant, compteur des 50 coups), et coup vers indice dans un espace d'actions fixe (8×8×73 comme AlphaZero), avec masque des coups légaux.
2. **Réseau** : ResNet (tronc de blocs résiduels, une tête politique et une tête valeur), taille réglable par config (petit en local, plus gros sur Kaggle).
3. **Données Lichess** : téléchargement et filtrage des parties (Elo, cadence), conversion PGN en exemples (position, coup joué, résultat) stockés en fichiers compacts.
4. **Pré-entraînement supervisé** : boucle d'entraînement avec perte politique + perte valeur, validation et checkpoints (4a, fait en local avec tiny), puis premier notebook Kaggle (4b).
5. **MCTS** : recherche guidée par le réseau (PUCT, bruit de Dirichlet à la racine, température), évaluation par lots avec perte virtuelle, réutilisation de l'arbre.
6. **Self-play** : génération de parties avec le MCTS, enregistrement des cibles (distribution de visites, résultat) dans un replay buffer.
7. **Boucle AlphaZero** : alternance self-play et entraînement, arène nouveau contre ancien modèle, reprise après coupure Kaggle.
8. **Évaluation et jeu** : mesure de force (contre Stockfish à niveau limité, estimation Elo), interface UCI pour jouer contre l'IA.

## Avancement

Règle : cette section ne garde que les décisions et les infos dont une future session a besoin.
Les détails (mesures complètes, listes de tests, historique) vont dans [docs/journal.md](docs/journal.md),
une section par étape, à compléter à la fin de chaque étape.

- [x] **Étape 0 — Mise en place** (2026-10-20)
  - Python 3.12.10 et Git 2.55 via winget ; venv avec torch 2.14.1+cpu, chess 1.11.2, numpy, tqdm, pytest.
  - `device.py` (choix du device) et `checkpoint.py` (reprise identique à un run d'une traite, testée).
- [x] **Étape 1 — Encodage** (2026-10-20)
  - `encoding.py` : tenseur `(18, 8, 8)` float32 du point de vue du camp au trait (Noirs = `board.mirror()`, retournement vertical seulement, pas de miroir gauche-droite à cause des roques). Plans : 0-5 mes pièces, 6-11 adverses, 12-15 roques, 16 en passant (seulement si la prise est légale), 17 compteur des 50 coups / 100. Pas d'historique ni de répétitions pour l'instant.
  - Coups : indice = `case_départ_retournée * 73 + plan` (4672) ; plans 0-55 dame, 56-63 cavalier, 64-72 sous-promotions ; promotion en dame = coup normal. API : `move_to_index(move, board)`, `index_to_move(index, board)`, `legal_move_mask(board)`.
  - Stockage compact : 67 octets `uint8` par position (`pack_board` / `unpack_batch`, ~270 Mo pour 4 M de positions). Les cibles (coup en int16, résultat en int8) seront stockées à côté à l'étape 3.
- [x] **Étape 2 — Réseau** (2026-10-20)
  - `model.py` : `ModelConfig.named("tiny" | "kaggle")` (lit `configs/*.toml`), `build_model`, `ChessNet` ; `forward` renvoie des logits bruts `(B, 4672)` (indice = `case_départ*73 + plan`, testé) et une valeur `(B,)` dans [-1, 1] (`tanh`) du point de vue du camp au trait. Paramètres : tiny 352 460 (4 × 64), kaggle 3 140 940 (10 × 128).
  - Perte : `policy_value_loss(logits, value, target_policy, target_value, legal_mask)` = entropie croisée avec coups illégaux masqués (cible = indice du coup ou distribution de visites) + MSE valeur. `masked_policy` donne les probabilités pour le MCTS.
  - Vitesse CPU (14 threads) : tiny ≈ 780 pos/s une par une, ≈ 5 000 en lots de 64+ ; kaggle ≈ 230 seule, ≈ 700 en lots. Le MCTS doit donc évaluer par lots, et le réseau kaggle est réservé au GPU.
  - `requires-python >= 3.11` (configs en TOML, `tomllib`).
- [x] **Étape 3 — Données Lichess** (2026-10-20, révisée le même jour)
  - `data.py` + `scripts/prepare_data.py` : base mensuelle Lichess 2026-09 lue **en flux HTTP + zstd** (jamais stockée, arrêt dès la cible ; `zstandard` ajouté aux dépendances), filtrage sur les en-têtes (Elo des deux joueurs ≥ 2000, pas de bullet = durée estimée < 180 s, Termination Normal/Time forfeit, ≥ 20 demi-coups, position de départ standard, **pas de joueur BOT** : `exclude_bots = true` par défaut, `--include-bots` pour l'annuler), config dans `configs/data.toml`.
  - Train/val **par partie** (`crc32(Site)`, 2 %). Shards `.npz` de 250 000 positions dans `data/<nom>/` : `boards (N,67) uint8`, `moves (N,) int16`, `results (N,) int8` (+1/0/−1 du camp au trait), `games (N,) int32`, **`value_valid (N,) uint8`** ; `manifest.json`. `data.load_split(dossier, "train"|"val")` charge tout en RAM.
  - **`value_valid` = 1 si la partie finit en « Normal », 0 si « Time forfeit »** (24 % des parties gardées, 29 % des positions). Raison : un résultat au drapeau dit qui a manqué de temps, pas qui avait la meilleure position (3,8 % de nulles contre 10,2 % en Normal), donc c'est une mauvaise cible de valeur ; le coup joué reste une bonne cible de politique. La perte de valeur ignore ces positions (étape 4a).
  - Extraction réelle `data/lichess_2000` : 686 052 parties lues, 924 BOT écartées, 38 962 gardées, **3 004 035 positions** (train 2 943 133, val 60 902 ; 71 % ont `value_valid = 1`), 225 Mo, 45 s. Détails, rejets et tests dans le journal.
- [x] **Étape 4a — Pré-entraînement supervisé, tiny en local** (2026-10-20)
  - `train.py` + `scripts/train.py --config tiny --out checkpoints/tiny_sup` : permutation par époque dérivée de `(seed, époque)` (le lot d'un pas ne dépend que du pas : reprise exacte sans état de lecture), AdamW (lr 1e-3 tiny / 2e-3 kaggle, wd 0,01 hors BatchNorm et biais, clip 1,0), montée linéaire puis cosinus jusqu'à 1 %, fp16 + GradScaler **seulement sur CUDA** (exercé sur Kaggle à l'étape 4b), paramètres dans la section `[train]` de `tiny.toml` / `kaggle.toml`.
  - **Perte** : politique sur toutes les positions (coups légaux seulement) + valeur **sur les positions `value_valid = 1` seulement** (`policy_value_loss(..., value_valid)` ; moyenne sur ces positions, gradient nul pour les autres, testé). Même règle pour la perte valeur de validation.
  - Les masques de coups légaux (python-chess, ~7 000 pos/s) sont calculés par un pool de processus (`num_workers`) ; lots identiques avec ou sans pool.
  - Suivi : `metrics.csv` (perte politique, perte valeur, top-1, top-3 ; lignes `train` = moyenne par fenêtre, `val` = 20 000 positions fixes, `val_full` = toute la validation à la fin), `scripts/plot_curves.py` (nécessite matplotlib, extra `plot`). `best.pt` = meilleure perte de validation, `last.pt` = reprise automatique (même commande relancée ; coupure simulée avec `--stop-after`). Reprise testée : mêmes poids (`torch.equal`) et même CSV qu'un run d'une traite.
  - **Résultat tiny, 5 000 pas (0,43 époque), 13 min, ~1 800 pos/s** : validation complète → perte politique 2,062, perte valeur 0,784 (référence « toujours 0 » : 0,92), **top-1 38,2 %, top-3 64,5 %** (hasard : ~3 %). Pas de surapprentissage, la courbe baisse encore : plus de pas aideraient. Courbes : `docs/img/tiny_sup_curves.png`. `best.pt` = pas 5 000.
  - `scripts/show_policy.py <checkpoint> [--fen ...] [--moves "e2e4 ..."]` : 5 meilleurs coups avec probabilités et valeur (position initiale : e4 50,8 %, d4 30,4 %, Nf3 6,7 %).
- [x] **Étape 4b — Pré-entraînement du réseau kaggle sur GPU** (2026-10-20) → **étape 4 terminée**
  - **Modèle livré : `checkpoints/kaggle_sup/best.pt`** (= `last.pt`, identiques : le dernier pas est le meilleur ; 37,8 Mo, hors git ; avec `metrics.csv`). Réseau `kaggle` (10 × 128, 3 140 940 paramètres), 92 000 pas de 1 024 = 4,00 époques, 3 h 47 sur Kaggle. Se charge avec `ModelConfig.named("kaggle")` ; c'est le point de départ du MCTS (étape 5) et du self-play. Courbes : `docs/img/kaggle_sup_curves.png`.
  - **Validation complète (475 048 positions)** : perte politique **1,407**, perte valeur **0,728** (référence « toujours 0 » : 0,899), **top-1 53,5 %, top-3 81,9 %** (tiny : 38,2 % / 64,5 %). La perte valeur de validation est plate depuis le pas ~60 000 (min 0,7266 au pas 65 000, fin 0,7284) : pas de remontée durable ; léger surapprentissage de la tête valeur (train 0,686 contre val 0,729, écart qui grandit à chaque époque) ; la politique progresse encore un peu (val 1,410).
  - Git : historique réécrit avec l'adresse noreply `345045991+Enz091@users.noreply.github.com` (nom d'auteur inchangé) ; remote `origin` = `https://github.com/Enz091/Project_chess.git`. Ne jamais mettre l'adresse personnelle dans un fichier.
  - Données : `prepare_data.py --months 2026-02 ... 2026-09 --per-source-positions 3000000` → `data/lichess_24m` : **24 035 177 positions** (train 23 560 129, val 475 048), 317 s ; chaque mois ne donne que ses ~650 000 premières parties (flux zstd non seekable). `data.extract` accepte une liste de sources. Dataset Kaggle : `data/kaggle/lichess_24m.zip` (256 Mo, fait par `make_kaggle_zip.py`).
  - Outils : `train.py` mesure débit, attente des données / masques et utilisation du GPU (colonnes de `metrics.csv`), `--max-minutes N` (arrêt propre avec `last.pt`), `mask_train_loss` (perte sans masque, **désactivée** : +0,6 % de débit seulement, apprentissage plus lent) ; `summarize_run.py` ; `notebooks/kaggle_train.ipynb` (reprise via la sortie du notebook ; ajouter `src` à `sys.path` après `pip install -e`). Les masques à l'entraînement ne ralentissent pas le GPU (7 970 pos/s, GPU 94 %, attente 0 %). Au démarrage Kaggle, ~8 min de chargement des données et des masques de validation.
  - `kaggle.toml` : `total_steps = 92000`, `val_positions = 100000`, warmup 1 000 pas puis cosinus jusqu'à 1 % (ne plus changer `total_steps` en cours de run). Le chemin fp16/GradScaler a été exercé sur GPU Kaggle.
- [x] **Étape 5 — MCTS** (2026-10-20)
  - `mcts.py` : `MCTS(evaluator, MCTSConfig.named("kaggle"), board)`, `search(N)` (ajoute exactement N simulations), `best_move(temperature)`, `visit_distribution()` (vecteur (4672,), cible du self-play), `advance(move)` pour les coups des deux camps (le sous-arbre devient la racine), `root_stats()`, `principal_variation()`, `check_tree()`. `NetEvaluator(model)` évalue un lot en un appel au réseau ; `policy_best_move` = réseau sans recherche ; `checkpoint.load_for_inference(path, device)` recharge un modèle.
  - **Signes** : la valeur du réseau est vue par le camp au trait dans la feuille ; `W`/`Q` d'une arête sont vus par le camp qui choisit ce coup ; la remontée change le signe avant d'ajouter la valeur à l'arête qui mène à la feuille, puis à chaque niveau. **FPU** = `node.q() − 0,2`, où `node.q()` = (valeur réseau du nœud + ΣW) / (1 + ΣN) = valeur pour le camp qui choisit dans ce nœud (= −Q de l'arête entrante, mais définie aussi à la racine). Testé, y compris par mutation.
  - Terminaux exacts : mat −1 ; pat, matériel insuffisant, 50 coups (≥ 100 demi-coups) et **triple répétition exacte** (avec l'historique de la partie) → 0. Perte virtuelle = compteur entier `VL`, entièrement retiré après chaque lot (testé). Dirichlet et température présents mais **désactivés par défaut** (`dirichlet_epsilon = 0`, `temperature = 0`), à activer pour l'étape 6.
  - Section `[mcts]` des configs : 200 simulations, `c_puct = 1.5`, **`batch_size = 8`**, `virtual_loss = 1`, `fpu_reduction = 0.2`, `dirichlet_alpha = 0.3`.
  - **Vitesse CPU (réseau kaggle)** : 220 simulations/s en lots de 1, **430 en lots de 8**, 540 en lots de 16, 670 en lots de 64 ; le réseau prend plus de 80 % du temps. Au-delà de 8, la perte virtuelle aplatit les visites (en lots de 64, la recherche retombe vers les priors). Sur GPU, il faudra plutôt des lots formés de plusieurs parties en parallèle.
  - **Résultats** : le MCTS trouve le mat du couloir (Ra8#, la politique jouait Ra7) et deux mats en deux (piège de Légal Bxf7+, la politique jouait Nxf7 ; mat étouffé Qg8+, la politique jouait Nf7+ ; serré à 200 simulations, net à 400). **Match MCTS-200 contre politique seule : 20-0, toutes par mat** (écart ≥ +317 Elo, borne à 95 %), 0,45 s par coup ; parties dans `docs/match_mcts200_vs_policy.pgn`.
  - Limite connue : un gain immédiat et un gain lointain valent tous deux +1, donc pas de préférence pour le mat le plus court (gains convertis lentement en finale).
- [ ] Étape 6 — Self-play
- [ ] Étape 7 — Boucle AlphaZero
- [ ] Étape 8 — Évaluation et jeu

## Commandes utiles

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\prepare_data.py --out data\lichess_2000 --target-positions 3000000   # --max-games 1000 pour un essai
.venv\Scripts\python.exe scripts\prepare_data.py --months 2026-02 2026-03 --per-source-positions 3000000 --out data\lichess_24m --workers 7   # plusieurs mois
.venv\Scripts\python.exe scripts\make_kaggle_zip.py data\lichess_24m --out data\kaggle\lichess_24m.zip   # archive à téléverser sur Kaggle
.venv\Scripts\python.exe scripts\train.py --config tiny --out checkpoints\tiny_sup   # relancer = reprendre ; --steps N pour un essai
.venv\Scripts\python.exe scripts\summarize_run.py checkpoints\tiny_sup --skip 1   # débit, attente des données, verdict
.venv\Scripts\python.exe scripts\train.py --config tiny --out checkpoints\essai --max-minutes 5 --no-mask-loss   # budget de temps, perte sans masque
.venv\Scripts\python.exe scripts\plot_curves.py checkpoints\tiny_sup   # -> curves.png
.venv\Scripts\python.exe scripts\show_policy.py checkpoints\tiny_sup\best.pt   # --fen "<FEN>" ou --moves "e2e4 e7e5"
.venv\Scripts\python.exe scripts\bench_inference.py --configs tiny   # vitesse d'inférence
.venv\Scripts\python.exe scripts\show_search.py checkpoints\kaggle_sup\best.pt --fen "<FEN>" --simulations 400   # recherche MCTS commentée
.venv\Scripts\python.exe scripts\bench_mcts.py   # simulations/s selon la taille de lot (réseau kaggle)
.venv\Scripts\python.exe scripts\match.py --pgn docs\match_mcts200_vs_policy.pgn   # MCTS-200 contre politique seule, 20 parties
.venv\Scripts\python.exe scripts\show_encoding.py --fen "<FEN>"
.venv\Scripts\python.exe scripts\smoke_resume.py --stop-after 60   # puis relancer sans l'option
```
