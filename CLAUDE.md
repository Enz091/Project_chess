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
  Le dépôt GitHub n'existe pas encore.

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
src/chesszero/   device.py, checkpoint.py, encoding.py, model.py   (+ mcts, selfplay, train à venir)
scripts/         points d'entrée (smoke_resume.py = démo reprise, show_encoding.py = démo encodage, bench_inference.py = vitesse du réseau)
configs/         tiny.toml (4 blocs × 64, CPU) et kaggle.toml (10 blocs × 128), section [model]
tests/           pytest
docs/            journal.md = détails, mesures et tests de chaque étape
notebooks/       notebook Kaggle (clone + lance un script)
data/            PGN Lichess et données prétraitées (ignoré par git)
checkpoints/     sauvegardes d'entraînement (ignoré par git)
```

## Feuille de route

0. **Mise en place** : venv, PyTorch CPU, python-chess, git, arborescence, détection du device, checkpoints avec reprise.
1. **Encodage** : plateau vers tenseur (plans de pièces, roques, en passant, compteur des 50 coups), et coup vers indice dans un espace d'actions fixe (8×8×73 comme AlphaZero), avec masque des coups légaux.
2. **Réseau** : ResNet (tronc de blocs résiduels, une tête politique et une tête valeur), taille réglable par config (petit en local, plus gros sur Kaggle).
3. **Données Lichess** : téléchargement et filtrage des parties (Elo, cadence), conversion PGN en exemples (position, coup joué, résultat) stockés en fichiers compacts.
4. **Pré-entraînement supervisé** : boucle d'entraînement avec perte politique + perte valeur, validation et checkpoints. Premier notebook Kaggle.
5. **MCTS** : recherche guidée par le réseau (PUCT, bruit de Dirichlet à la racine, température), évaluation par lots.
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
- [ ] Étape 3 — Données Lichess
- [ ] Étape 4 — Pré-entraînement supervisé
- [ ] Étape 5 — MCTS
- [ ] Étape 6 — Self-play
- [ ] Étape 7 — Boucle AlphaZero
- [ ] Étape 8 — Évaluation et jeu

## Commandes utiles

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\bench_inference.py --configs tiny   # vitesse d'inférence
.venv\Scripts\python.exe scripts\show_encoding.py --fen "<FEN>"
.venv\Scripts\python.exe scripts\smoke_resume.py --stop-after 60   # puis relancer sans l'option
```
