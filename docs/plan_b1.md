# Plan B1 : un réseau plus gros que `kaggle_sup` (proposition du 2026-11-08, rien d'implémenté)

Objectif de la voie B (`docs/options_apres_session3.md`) : un réseau de référence plus fort que `kaggle_sup`
(10 blocs × 128 canaux, 3,1 M de paramètres, 2423 ± 26 au protocole v2). Deux façons de grossir le réseau sont
comparées ici. B1 est indépendant de B2 (valeur entraînée sur Stockfish, en cours sur le PC) : si B2 réussit, sa
recette d'affinage s'appliquera ensuite au réseau B1 avec les mêmes étiquettes.

## Chiffres de départ (mesurés)

- Pré-entraînement de `kaggle_sup` sur T4 : 92 000 pas de 1 024 = 4 passages sur 23,6 M de positions,
  7 970 positions/s, GPU à 94 % (limité par le GPU), **3 h 47** + ~8 min de chargement.
- Données `lichess_24m` : 8 mois (2026-02 à 2026-09), les ~650 000 premières parties de chaque mois (le flux zstd
  ne permet pas de sauter plus loin), **1,74 Go téléchargés, 317 s** sur le PC, 1,80 Go sur disque, archive
  Kaggle de **256 Mo**.
- La tête valeur de `kaggle_sup` surapprend déjà (0,686 en entraînement contre 0,729 en validation) et sa
  politique progressait encore à la fin : plus de données et plus de capacité vont dans le bon sens.
- Coût du réseau ailleurs : self-play sur Kaggle, réseau ≈ 20 % du temps d'un processus (GPU à 24 %) ; sur le
  PC, réseau > 80 % du temps du MCTS ; mesure v2 de `kaggle_sup` : 2 h 20 ; arène locale de 200 parties
  (essai B2) : ~1 h 15 à 1 h 30.
- Calcul d'un réseau ∝ blocs × canaux² : 15 × 128 = ×1,5 ; 12 × 160 = ×1,9.

## Les deux options

### (a) Approfondir `kaggle_sup` : 10 → 15 blocs × 128, puis affiner
- **Principe** : insérer 5 blocs résiduels (un après chaque paire de blocs existants) dont le second BatchNorm a
  γ = β = 0. Un bloc calcule relu(x + BN(conv(relu(BN(conv(x)))))) ; avec ce BN nul il renvoie relu(x) = x (x sort
  déjà d'un relu) : **le réseau approfondi est exactement `kaggle_sup` au départ** (à vérifier par un test, à la
  sortie près). Le gradient passe par γ, qui s'ouvre progressivement (technique du « zero-init residual »). Puis
  affinage complet (pas seulement les nouveaux blocs), taux plus bas qu'un entraînement de zéro (pic ~5e-4, cosinus).
- **Données** : les 24 M actuelles suffisent pour un premier essai (**aucun nouveau Dataset**) ; avec 48 M, mêmes
  données que (b), voir plus bas.
- **Quota Kaggle** : ~5 300 positions/s (×1,5) ; 46 000 pas (2 passages sur 24 M ou 1 sur 48 M) ≈ **2 h 30** + chargement.
  **Il faut un témoin** : le même affinage sur `kaggle_sup` sans blocs ajoutés (~1 h 40), sinon on ne sait pas si le
  gain vient de la profondeur ou simplement de 46 000 pas d'entraînement de plus (la politique progressait encore).
  Total **~4 h 20**.
- **Si l'on reprend le RL** : self-play −9 % au plus (borne haute : une partie du temps « réseau » est du travail CPU
  qui ne grossit pas), arène ~14 → ~15,5 min.
- **Mesure v2 sur le PC** : ~0,2 + 0,8 × 1,5 = ×1,4 → **~3 h 15** ; arène locale ~1 h 45.
- **Gain attendu** : **+10 à +40 Elo, confiance faible**. Le réseau garde ses 128 canaux (la largeur limite ce que
  chaque couche représente), part d'un point déjà convergé et ne reçoit que peu de pas pour exploiter la capacité
  ajoutée ; une partie du gain éventuel viendrait de l'entraînement prolongé, que le témoin mesure à part.

### (b) Un réseau ~2 fois plus lourd entraîné de zéro : 12 blocs × 160
- **Principe** : même recette que `kaggle_sup` (AdamW, pic 2e-3, montée de 1 000 pas puis cosinus jusqu'à 1 %,
  fp16, 92 000 pas de 1 024), réseau 12 × 160 (~5,8 M de paramètres, ×1,9 en calcul), **sur 48 M de positions**
  (2 passages au lieu de 4 : la tête valeur surapprend moins).
- **Données (48 M)** : `lichess_24m` + **8 mois de plus** (2025-06 à 2026-01), même méthode et mêmes filtres (Elo
  ≥ 2000 des deux joueurs, pas de bullet ni de BOT, `value_valid` = 0 au drapeau), extraits en une fois avec
  `prepare_data.py --months 2025-06 … 2026-09 --per-source-positions 3000000 --out data/lichess_48m` (16 mois).
  La validation (2 % des parties par `crc32(Site)`) contient exactement celle de `lichess_24m`. Coût : **~1,7 Go de
  téléchargement de plus, ~11 min** pour les 16 mois sur le PC, 3,6 Go sur disque ; archive `make_kaggle_zip.py`
  **~510 Mo à téléverser** (nouveau Dataset privé `lichess-48m`, quelques minutes selon la connexion) ; en RAM sur
  Kaggle : ~3,6 Go (sur ~30). Le mois de janvier 2026 a servi au livre d'ouvertures du self-play : sans effet pour
  un entraînement supervisé (les ouvertures v2 n'étaient pas exclues des données de `kaggle_sup` non plus ; même règle).
- **Quota Kaggle** : ~4 100 positions/s (×1,9, entraînement limité par le GPU) → **~6 h 25 + ~10 min** de
  chargement : une session (12 h max ; la reprise depuis `last.pt` existe en cas de coupure). ~22 % du quota d'une
  semaine.
- **Si l'on reprend le RL** : self-play −15 % au plus, GPU de ~24 % à ~45 % ; arène ~14 → ~16,5 min.
- **Mesure v2 sur le PC** : ×1,72 → **~4 h** (une soirée ou une nuit, sur secteur) ; arène locale ~2 h 10.
- **Gain attendu** : **+30 à +80 Elo à 200 simulations, confiance moyenne**. Pour : capacité ~2 fois plus grande,
  deux fois plus de données, politique et valeur encore en progrès à la fin de `kaggle_sup`. Contre : la
  prédiction de coups humains plafonne (vers 55-58 % de top-1), et imiter mieux des joueurs 2000+ ne rend pas
  forcément le jeu plus fort ; aucune mesure de ce projet ne chiffre directement l'effet de la taille.

## Comparaison

| | (a) approfondir 15 × 128 | (b) 12 × 160 de zéro |
|---|---|---|
| Nouveau Dataset | non (24 M suffisent) | oui : 48 M, ~11 min sur le PC, ~510 Mo à téléverser |
| Quota Kaggle | ~4 h 20 (avec le témoin indispensable) | ~6 h 35 |
| Self-play / arène si RL repris | −9 % / ~15,5 min | −15 % / ~16,5 min |
| Mesure v2 sur le PC | ~3 h 15 | ~4 h |
| Gain attendu | +10 à +40 (confiance faible) | +30 à +80 (confiance moyenne) |
| Risque principal | gain trop petit pour être vu (v2 : ±44), attribution floue | gain absorbé par la limite des données humaines |
| Développement | insertion des blocs + test d'identité, ~½ jour | config + données + notebook, ~½ jour |

## Recommandation : (b), le réseau 12 × 160 entraîné de zéro sur 48 M de positions

- C'est l'option qui a une chance réaliste de dépasser le seuil de significativité de la mesure v2 (+44 Elo) : le
  gain attendu de (a) est du même ordre que ce seuil, donc probablement indétectable, et il faudrait un témoin pour
  l'attribuer.
- Le coût supplémentaire est modeste : ~2 h de quota de plus que (a), un Dataset préparé en ~11 min, une mesure v2
  plus longue de ~45 min.
- Le résultat est propre : même recette que `kaggle_sup`, seuls la taille et les données changent ; la recette B2
  s'y appliquera ensuite telle quelle si B2 réussit (les 1 M d'étiquettes Stockfish servent sans rien refaire ;
  affinage ~2 h 30 sur le PC pour ce réseau).
- (a) reste le repli si le quota manque.

### Critères de succès proposés (à fixer avant le lancement)
1. **Santé du pré-entraînement** (Kaggle) : débit ≥ 3 000 positions/s après 2 000 pas (sinon arrêt et révision,
   pour ne pas brûler le quota) ; pas de divergence (perte de validation finie et en baisse au pas 10 000).
2. **Politique et valeur supervisées meilleures que `kaggle_sup`**, mesurées avec le même outil et le même
   sous-ensemble que B2 (validation Lichess, 50 000 positions fixes de `lichess_24m` : `kaggle_sup` top-1 53,70 %,
   perte de politique 1,4049, perte de valeur 0,7267) : **top-1 ≥ 54,2 % ET perte de politique ≤ 1,3949** (−0,01).
   Sinon : échec (le réseau plus gros n'apprend pas plus des données humaines), pas d'arène.
3. **Arène locale contre `kaggle_sup`** (200 parties, 100 ouvertures fixes de l'arène ancre, 200 simulations, tables
   WDL) : **mesure v2 seulement si ≥ +20 Elo**, et avec l'accord de l'auteur.
4. **Mesure v2** (`--like kaggle_sup`) : **gain retenu seulement au-delà de +44 Elo** par rapport à 2423 ± 26 ; le
   réseau devient alors la nouvelle référence (et la nouvelle ancre si le RL reprend).

### Ce qu'il faudra faire après l'accord (aucun quota avant)
- `configs/kaggle_l.toml` (12 × 160, même `[train]`) ; extraction des 16 mois et archive ; essai local de quelques
  pas sur CPU avec la nouvelle config ; notebook `kaggle_train.ipynb` avec ce config et le Dataset `lichess-48m`.
- `match_stockfish.py` : le bench de départ attend ≈ 0,80 s par coup (réseau 10 × 128) ; avec un réseau ×1,9 il
  avertira à tort. Prévoir une référence de vitesse propre au réseau mesuré (sans toucher au protocole v2, qui fixe
  les simulations et le temps de Stockfish, pas la vitesse de l'IA).
