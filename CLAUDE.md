# Projet_chess — IA d'échecs façon AlphaZero

IA d'échecs en Python (PyTorch + python-chess) : un réseau de neurones à deux têtes
(**politique** = probabilité de chaque coup, **valeur** = qui gagne) guide une recherche
**MCTS**. Le réseau est d'abord **pré-entraîné en supervisé** sur des parties Lichess, puis
amélioré par **self-play**.

L'auteur veut comprendre l'architecture globale : expliquer chaque choix avant de l'appliquer,
avancer étape par étape, et **mettre à jour ce fichier (section Avancement) à la fin de chaque étape**.

## Environnement

- **Local (développement / tests)** : Windows 11, Intel Iris Xe, pas de GPU NVIDIA → CPU.
  Python 3.12, venv dans `.venv/`, PyTorch **version CPU**.
- **Entraînements réels** : Kaggle (GPU), le code arrive par GitHub (`git clone`).
- Git : branche `main`, auteur `345045991+Enz091@users.noreply.github.com` (config locale au dépôt).
  Le dépôt GitHub n'existe pas encore.

### Installation locale
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
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
src/chesszero/   device.py, checkpoint.py   (+ encoding, model, mcts, selfplay, train à venir)
scripts/         points d'entrée (smoke_resume.py = démo reprise)
configs/         hyperparamètres (local = petits, kaggle = gros)
tests/           pytest
notebooks/       notebook Kaggle (clone + lance un script)
data/            PGN Lichess et données prétraitées (ignoré par git)
checkpoints/     sauvegardes d'entraînement (ignoré par git)
```

## Feuille de route

0. **Mise en place** : venv, PyTorch CPU, python-chess, git, arborescence, détection du device, checkpoints avec reprise.
1. **Encodage** : plateau vers tenseur (plans de pièces, trait, roques, en passant, répétitions), et coup vers indice dans un espace d'actions fixe (8×8×73 comme AlphaZero), avec masque des coups légaux.
2. **Réseau** : ResNet (tronc de blocs résiduels, une tête politique et une tête valeur), taille réglable par config (petit en local, plus gros sur Kaggle).
3. **Données Lichess** : téléchargement et filtrage des parties (Elo, cadence), conversion PGN en exemples (position, coup joué, résultat) stockés en fichiers compacts.
4. **Pré-entraînement supervisé** : boucle d'entraînement avec perte politique + perte valeur, validation et checkpoints. Premier notebook Kaggle.
5. **MCTS** : recherche guidée par le réseau (PUCT, bruit de Dirichlet à la racine, température), évaluation par lots.
6. **Self-play** : génération de parties avec le MCTS, enregistrement des cibles (distribution de visites, résultat) dans un replay buffer.
7. **Boucle AlphaZero** : alternance self-play et entraînement, arène nouveau contre ancien modèle, reprise après coupure Kaggle.
8. **Évaluation et jeu** : mesure de force (contre Stockfish à niveau limité, estimation Elo), interface UCI pour jouer contre l'IA.

## Avancement

- [x] **Étape 0 — Mise en place** (2026-10-20)
  - Python 3.12.10 et Git 2.55 installés via winget. Venv avec torch 2.14.1+cpu, chess 1.11.2, numpy, tqdm, pytest.
  - `device.py` et `checkpoint.py` écrits. 3 tests verts, dont « reprise = run d'une traite ».
  - `scripts/smoke_resume.py` : interruption à l'étape 60 puis reprise automatique vérifiée.
- [ ] Étape 1 — Encodage
- [ ] Étape 2 — Réseau
- [ ] Étape 3 — Données Lichess
- [ ] Étape 4 — Pré-entraînement supervisé
- [ ] Étape 5 — MCTS
- [ ] Étape 6 — Self-play
- [ ] Étape 7 — Boucle AlphaZero
- [ ] Étape 8 — Évaluation et jeu

## Commandes utiles

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\smoke_resume.py --stop-after 60   # puis relancer sans l'option
```
