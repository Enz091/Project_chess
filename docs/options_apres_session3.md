# Options après la session de boucle 3 (document de décision, 2026-11-07)

Rien n'est implémenté ni lancé : ce document compare deux voies pour la suite et se termine par une recommandation.
Les chiffres viennent du journal (`docs/journal.md`, sections des sessions de boucle 1 à 3) et de l'analyse du
2026-11-07 (`docs/analyse_session3/`).

## 1. Où on en est

- **Référence** : `kaggle_sup` (10 blocs × 128, supervisé sur 24 M de positions Lichess 2000+), **2423 ± 26** au
  protocole v2 (échelle Stockfish bridé, 200 simulations, 1 280 parties).
- **Boucle AlphaZero : 13 candidats en 3 sessions (~25 h de GPU Kaggle), aucun meilleur que `kaggle_sup`.**
  Sessions 1-2 (générateur = `kaggle_sup`) : candidats entre −35 et −5 contre l'ancre, tous compatibles avec
  −20 ± 17. Session 3 (règle « los » 0,20, générateur = dernier accepté) : 4 générations acceptées, contre l'ancre
  +9 (biaisé), −21, −23, **−74 (−117 à −34)** ; critère d'échec fixé à l'avance atteint, boucle arrêtée sous cette
  forme.
- **Ce que l'analyse du 2026-11-07 a établi** (détails dans le journal) :
  1. L'hypothèse d'auto-renforcement (politique et recherche qui se concentrent, gaffes en position gagnante)
     **n'est pas soutenue** : la politique s'est surtout resserrée avant toute acceptation (entropie 1,43 → 1,36
     pour le candidat 6, à −5 Elo seulement), à peine ensuite (1,36 → 1,33) ; l'entropie des visites d'une
     recherche à 200 simulations est plate de génération en génération ; sur 250 positions gagnantes fixes, les
     générations 2 à 4 jouent aussi bien ou mieux que `kaggle_sup` (−4 à −10 cp) ; avec un juge unique, elles ne
     convertissent pas plus mal (15,8 % contre 16,4 %). Le « 18 % contre 2 % » de l'indicateur de conversion était un
     effet de confiance.
  2. Ce qui est observé à la place : une **dérive** loin de `kaggle_sup` (coups préférés différents 12 → 16 %,
     top-1 Lichess 53,7 → 52,8 %, perte de politique Lichess +0,047 : le garde-fou G1 rejette à +0,05), une
     **non-transitivité** (chaque génération fait jeu égal avec la précédente, somme des arènes A = +1 Elo, mais
     recule contre l'ancre) et, dans les vraies parties de la génération 4, **+40 % de gaffes au milieu de partie**
     (57,5 ‰ contre 41,2 ‰) et beaucoup moins de positions gagnantes obtenues (72 contre 110).
  3. Les cibles du self-play se resserrent à chaque générateur (entropie débruitée 0,906 → 0,843), lentement.
- **La faiblesse connue de `kaggle_sup` lui-même** (parties v2) : gaffes tactiques au milieu de partie (13 défaites
  contre 1600-1900, dont 11 après avoir été à +1 ou mieux) ; les finales ≤ 5 pièces sont réglées par les tables.

Échelle de calcul, pour fixer les idées : la boucle a produit ~2,4 M de positions de self-play (~23 000 parties à
200 simulations). AlphaZero en a joué ~44 M de parties à 800 simulations ; Leela Chess Zero a eu besoin de millions de
parties pour dépasser un niveau humain fort. À notre échelle, le self-play ne fournit pas assez de signal pour battre
un point de départ supervisé déjà fort, et chaque génération perd un peu de ce que le supervisé savait.

### Coûts de référence (mesurés)

| poste | coût actuel |
|---|---|
| Quota Kaggle | 30 h de GPU par semaine ; session ≤ 12 h (11 h utiles) |
| Pré-entraînement 10 × 128 (92 000 pas de 1 024, T4 fp16, GPU à 94 %) | 3 h 47 + ~8 min de chargement |
| Self-play sur Kaggle (3 processus × 64 parties) | ~4 400 simulations/s, ~100 000 positions en 77 min ; GPU 24 %, réseau ~20 % du temps d'un processus |
| Arène 200 parties sur Kaggle | ~14 min |
| Mesure v2 sur le PC (`--like kaggle_sup`, 640 parties) | 2 h 20 (réseau 10 × 128 sur CPU ; écart minimal significatif contre la référence : ±44 Elo) |
| Arène 200 parties sur le PC (200 simulations) | ~2 h (estimation : ~560 simulations/s) |
| Stockfish profondeur 14 sur le PC (10 processus) | ~125 positions/s (56 306 en 7,5 min le 2026-11-07) |
| Entraînement du réseau 10 × 128 sur le PC (CPU) | ~165 positions/s (435 pas de 1 024 ≈ 45 min) |

## 2. Voie A — réparer le RL

Principe commun : garder la boucle (self-play, tour, arène) en empêchant la dérive, et/ou en rendant les cibles plus
fortes que le réseau. Limite commune : **un test réel demande plusieurs générations, donc du quota Kaggle** ; le PC
ne joue que ~650 simulations/s (une génération de 100 000 positions ≈ 8 à 9 h), il ne peut tester qu'un tour hors
ligne sur les tampons existants.

### A1. Régularisation KL vers la politique de `kaggle_sup` (comme en RLHF)
- **Idée** : perte de politique + β · KL(π_θ ‖ π_sup) sur les positions de self-play (ou, ce qui revient presque au
  même, cible = (1 − λ) · cible MCTS + λ · π_sup). L'infrastructure existe en partie : l'entraîneur sait déjà calculer
  les priors d'un réseau sur les anciens shards (`sp_train/policy_priors/`).
- **Gain attendu** : arrête la dérive mesurée (coups préférés, Lichess) — **confiance moyenne** pour cet effet
  protecteur. Gain d'Elo au-delà de `kaggle_sup` : 0 à +20, **confiance faible** : aucun des 13 candidats n'a montré
  de progrès, même ceux qui dérivaient peu (candidat 6 : −5).
- **Coût** : développement ½ à 1 jour ; PC ~1 h (tour hors ligne) ; Kaggle : 1 à 2 sessions de 11 h pour voir la
  dynamique sur 3 à 5 générations.
- **Risque** : β trop fort = le réseau reste `kaggle_sup` (rien n'est appris) ; trop faible = même dérive. Un réglage
  de plus à trouver avec peu de mesures.
- **Expérience minimale sur le PC** : rejouer le tour 12 (même fenêtre du tampon de la session 3, même nombre de
  pas) depuis g3 avec β ∈ {0, 0,1, 0,3} ; mesurer sur les parties mises de côté l'accord avec `kaggle_sup`, la perte
  de politique Lichess, la valeur contre Stockfish (`eval_value.npz`), puis une arène locale de 200 parties contre
  `kaggle_sup` pour la meilleure variante (~2 h). Critère : dérive Lichess divisée par 2 sans perte de la valeur.

### A2. Plancher d'entropie
- **Idée** : pénaliser l'entropie de la politique sous un seuil (par exemple celle de `kaggle_sup`, 1,42).
- **Gain attendu** : ~0, **confiance moyenne** : l'analyse montre que la politique et la recherche ne se concentrent
  presque plus après le candidat 6 ; le resserrement principal (cibles débruitées) n'a pas coûté d'Elo. Rien ne
  justifie cette option.
- **Coût** : ½ jour ; mêmes tests que A1. **Risque** : réintroduit l'aplatissement de la session 1 (cibles
  bruitées), qui était nuisible.
- **Expérience minimale** : aucune recommandée ; si on la garde, même protocole que A1.

### A3. Part humaine augmentée (25 % → 50 %)
- **Idée** : plus de positions Lichess dans chaque lot, pour ancrer la politique et la valeur.
- **Gain attendu** : effet protecteur proche de A1 mais plus grossier (agit sur les données, pas sur la sortie) ;
  Elo au-delà de `kaggle_sup` 0 à +10, **confiance faible**. Divise par ~1,5 le signal de self-play par pas.
- **Coût** : réglage (`human_fraction`) ; PC ~1 h ; Kaggle comme A1.
- **Risque** : faible ; ralentit d'autant l'apprentissage par le self-play.
- **Expérience minimale** : comme A1, avec la part humaine à la place de β.

### A4. Acceptation jugée contre l'ancre plutôt que contre l'accepté
- **Idée** : remplacer l'arène A par l'arène ancre (même coût : 200 parties) ; accepter si le candidat n'est pas
  plus faible que `kaggle_sup` (ou que le meilleur accepté mesuré contre lui).
- **Gain attendu** : supprime le défaut de non-transitivité observé (la règle actuelle a accepté 4 générations dont
  la somme des écarts « contre le précédent » vaut +1 Elo, pendant que l'Elo contre l'ancre tombait à −74).
  **Confiance élevée** pour l'effet protecteur. Mais parmi les 11 candidats mesurés contre l'ancre, **seul le
  candidat 7 aurait passé une règle « au moins aussi bon que `kaggle_sup` »** (+9, P(meilleur) 68 % ; tous les autres
  ≤ 40 %) : on retombe dans le régime des sessions 1-2 (générateur presque figé, plateau vers −20). Aucun gain d'Elo
  attendu seul.
- **Coût** : petit (la règle et l'arène ancre existent) ; quota inchangé par génération.
- **Risque** : sélection biaisée vers l'ancre (le gagnant de 200 parties « sur-mesure » contre `kaggle_sup`) ;
  boucle figée.
- **Expérience minimale sur le PC** : `diagnose_session.py gates` avec les 13 arènes (simulation des règles), sans
  quota. À combiner avec une option qui produit du progrès ; seule, elle ne sert qu'à ne pas reculer.

### A5. Plus de simulations pour les cibles (800 au lieu de 200)
- **Idée** : c'est le levier central d'AlphaZero (la recherche doit être nettement plus forte que le réseau). Variante
  moins chère : jouer à 200 simulations et faire 800 simulations sur ~25 % des coups enregistrés (coups « complets »
  façon KataGo ; les parties ne sont pas affaiblies, contrairement à la « playout cap randomization » écartée en 6b).
- **Gain attendu** : cibles plus fortes (800 contre 200 simulations ≈ +100 à +200 Elo de recherche, estimation non
  mesurée) ; effet sur l'apprentissage : inconnu, **confiance faible**. Rien dans l'analyse ne montre que les cibles
  à 200 simulations sont le facteur limitant (la perte de force vient de la dérive et du milieu de partie).
- **Coût** : self-play ×4 (×1,75 pour la variante 25 %) : ~5 h (ou ~2 h 15) pour 100 000 positions, soit 1 à 2
  générations par session de 11 h au lieu de 4 à 6. Développement : 0 (tout à 800) à 1 jour (variante).
- **Risque** : quota multiplié pour un effet incertain ; moins de générations, donc encore moins de mesures.
- **Expérience minimale sur le PC** : sur 500 positions des parties mises de côté, comparer la perte Stockfish du
  coup le plus visité et la distance entre cibles à 200 et 800 simulations (`selfref` avec `--simulations 800`,
  ~1 h) : si les cibles à 800 ne choisissent pas des coups nettement meilleurs (> 5 cp), l'option ne vaut pas son quota.

### A6. Parties contre l'ancre (ou une « ligue ») dans le self-play — justifiée par la non-transitivité
- **Idée** : une part des parties de self-play (par exemple 30 %) jouée entre l'accepté et `kaggle_sup` (ou des
  générations précédentes), comme les ligues d'AlphaStar ; le code de l'arène joue déjà deux réseaux.
- **Gain attendu** : le réseau voit ses propres faiblesses face à l'ancre au lieu de faire jeu égal avec lui-même ;
  **confiance faible à moyenne** pour l'effet protecteur, faible pour un gain net.
- **Coût** : 1 à 2 jours (self-play à deux réseaux, enregistrement des cibles des deux camps ou d'un seul) ; quota
  inchangé par position.
- **Risque** : complexité ; les parties contre `kaggle_sup` ramènent la cible vers ses coups (effet proche de A1).
- **Expérience minimale** : pas de test PC crédible (il faut une génération complète) ; 1 session Kaggle.

### A7. Moins de réutilisation par génération, garde-fou de dérive permanent — justifiés par la dérive
- **Idée** : `reuse` 4 → 2 (moins de pas par position nouvelle, donc moins d'ajustement à ses propres parties) et le
  garde-fou G1 (politique Lichess > supervisé + 0,05 → rejet) appliqué à toutes les générations, plus seulement aux
  5 premières (la génération 4 était à +0,047).
- **Gain attendu** : ralentit la dérive ; **confiance moyenne** pour cet effet, faible pour un gain d'Elo.
- **Coût** : réglages ; aucun développement notable.
- **Risque** : faible ; G1 permanent aurait sans doute arrêté la ligne à la génération 5.
- **Expérience minimale sur le PC** : rejouer le tour 12 avec `reuse` 2 (comme A1).

**Bilan de la voie A** : les options justifiées par l'analyse (A1, A4, A6, A7) sont **protectrices** : elles
empêcheraient de reculer sous `kaggle_sup`, aucune ne donne de raison de croire qu'on le dépassera à notre échelle de
calcul. Le seul levier « productif » (A5) multiplie le quota pour un effet non démontré.

## 3. Voie B — gagner en force par l'apprentissage supervisé

Principe : améliorer le réseau de départ lui-même ; chaque gain se mesure avec le protocole v2 (`match_stockfish.py run
--like kaggle_sup`), sur le PC, sans quota. Un candidat se trie d'abord par une arène locale de 200 parties contre
`kaggle_sup` (~2 h, résolution ~±40 Elo), puis la mesure v2 (2 h 20 pour 10 × 128 ; écart significatif ±44 Elo).

### B1. Réseau plus gros
- **Idée** : passer de 10 × 128 (3,1 M de paramètres) à ~2 fois le calcul (12 × 160 : ~5,8 M) ou ~3,4 fois
  (15 × 192 : ~10 M). Variante économique : **approfondir `kaggle_sup`** en insérant des blocs résiduels initialisés
  à l'identité (dernier BatchNorm à γ = 0 : même fonction au départ), puis poursuivre l'entraînement (~1 époque) ;
  l'élargissement (128 → 160 canaux) ne se fait pas aussi simplement.
- **Effet sur le débit** (calcul du réseau proportionnel à blocs × canaux²) :
  - pré-entraînement sur T4 (déjà limité par le GPU) : ×1,9 → ~7 h 10 ; ×3,4 → ~13 h (2 sessions, la reprise
    existe) ; 20 × 256 (×8) → ~30 h, tout le quota d'une semaine. Approfondissement 15 × 128 (×1,5), 1 époque :
    ~1 h 30 ;
  - self-play (si l'on reprend le RL un jour) : le réseau prend ~20 % du temps d'un processus et le GPU est à 24 %,
    donc au plus −15 % de débit à ×1,9 et −30 % à ×3,4 (bornes hautes : une partie de ces 20 % est du travail CPU qui
    ne grossit pas) ; le GPU sature vers ×4 ;
  - **sur le PC, tout ralentit presque en proportion** (le réseau fait > 80 % du temps du MCTS sur CPU) : mesure v2
    ~4 h à ×1,9, ~7 h à ×3,4 (une nuit, sur secteur). C'est le vrai coût récurrent de cette option.
- **Données** : la tête valeur de `kaggle_sup` surapprend déjà (0,686 en entraînement contre 0,729 en validation) ;
  un réseau plus gros en demande davantage. Extraire 8 mois de plus (~24 M de positions, ~5 min sur le PC, archive
  ~250 Mo) et garder 92 000 pas (2 passages sur 48 M de positions au lieu de 4 sur 24 M) ne coûte rien de plus en
  quota.
- **Gain attendu** : +30 à +100 Elo à 200 simulations, **confiance moyenne** (la politique de validation baissait
  encore à la fin du pré-entraînement, signe d'un réseau limité par sa taille ; le passage tiny → kaggle avait
  donné +15 points de top-1 ; le suivant donnera beaucoup moins, la prédiction de coups humains plafonne vers 55-58 %).
- **Coût** : développement ½ jour (config, approfondissement : 1 jour) ; quota 1,5 à 13 h ; PC : mesures plus lentes.
- **Risque** : gain absorbé par la limite des données humaines (on imite mieux des joueurs à 2000+, pas forcément on
  joue mieux) ; mesures plus longues.
- **Expérience minimale** : pas de test PC crédible pour l'entraînement (le CPU entraîne à ~165 positions/s). Le
  moins cher : approfondissement 15 × 128 + 1 époque sur Kaggle (~1 h 30 de quota), arène locale, puis v2.

### B2. Valeur entraînée sur des évaluations Stockfish plutôt que sur les résultats
- **Idée** : cible de valeur = espérance de Stockfish (`UCI_ShowWDL` : (G − P) / 1000, déjà utilisée par
  `eval_value_target.py label`), ou un mélange λ · Stockfish + (1 − λ) · résultat. Sources possibles : (i) positions
  de `lichess_24m` notées sur le PC (profondeur 12 : ~250 positions/s estimées, 1 M en ~1 h 10, 5 M en ~5 h 30 ;
  profondeur 14 : 125/s) ; (ii) les évaluations `%eval` présentes dans une partie des PGN Lichess (part et profondeur
  **à vérifier**) ; (iii) la base d'évaluations publiée par Lichess (**à vérifier** : contenu, licence, distribution
  des positions, qui n'est pas celle de nos parties). Les positions « Time forfeit » (29 %, sans cible de valeur
  aujourd'hui) en reçoivent une.
- **Gain attendu** : la valeur de `kaggle_sup` est bruitée (résultats de parties humaines) et prudente (|v| 0,35
  contre 0,54 pour Stockfish) ; une valeur qui « voit » la tactique devrait aider la recherche là où l'IA perd (gaffes
  au milieu de partie). Mais **une mise en garde mesurée** : en session 2, une valeur nettement mieux calibrée contre
  Stockfish (MSE 0,194 → 0,176) n'a rien changé à la force (hybrides politique/valeur non significatifs). Ici
  l'écart serait bien plus grand (entraînement direct sur Stockfish). Estimation +20 à +80 Elo, **confiance
  faible à moyenne**, avec une vraie possibilité de 0.
- **Coût** : développement 1 jour (lecture de la cible dans `data.py` / `train.py`, option de mélange) ; PC : 1 à
  6 h d'étiquetage + 2 h d'entraînement par essai ; quota : 0 pour l'essai, ~4 h pour un vrai pré-entraînement.
- **Risque** : on distille une partie de Stockfish (le projet s'éloigne de l'esprit « zéro », comme déjà avec les
  parties humaines) ; profondeur 12-14 encore faillible ; surapprentissage si peu de positions.
- **Expérience minimale sur le PC (aucun quota)** : noter 1 M de positions d'entraînement de `lichess_24m`
  (profondeur 12, ~1 h 10), affiner `kaggle_sup` 1 000 pas de 1 024 (politique sur le coup joué comme aujourd'hui,
  valeur sur Stockfish, ~1 h 45), mesurer la valeur contre Stockfish sur `eval_value.npz` et la perte de politique
  Lichess (inchangée attendue), arène locale de 200 parties contre `kaggle_sup` (~2 h) ; si ≥ +20 Elo, mesure v2
  (2 h 20). **Total ~7 h de PC, en deux soirées.** Variante à tester dans le même cadre si la valeur seule ne suffit
  pas : ajouter le meilleur coup de Stockfish comme cible de politique partielle sur les positions du milieu de
  partie (vise directement les gaffes tactiques, mais coûte un MultiPV ou un second passage).

### B3. Données Lichess de meilleur niveau
- **Idée** : filtres plus stricts (les deux joueurs ≥ 2200 ou ≥ 2400, parties rapides ou lentes seulement), en lisant
  davantage de mois ; ou pondérer les positions par l'Elo.
- **Gain attendu** : politique plus proche de bons coups ; 0 à +50 Elo, **confiance faible** (des joueurs de blitz
  à 2400 gaffent aussi ; le volume baisse fortement, donc plus de mois à lire). La part des parties gardées à 2400+
  est **à mesurer** (`prepare_data.py --max-games`).
- **Coût** : développement ~0 (filtres de `configs/data.toml`) ; PC : lecture en flux d'un mois entier (fichier de
  ~30 Go compressé, ~1 h 30 à 2 h par mois, bande passante) ; quota : ré-entraînement ou affinage ~2 à 4 h.
- **Risque** : trop peu de données pour un réseau plus gros ; biais vers les ouvertures des joueurs forts.
- **Expérience minimale sur le PC** : extraire ~1 M de positions 2400+, affiner `kaggle_sup` (~1 h 45), arène locale
  contre `kaggle_sup` (~2 h).

## 4. Synthèse

| option | gain attendu (Elo) | confiance | développement | PC | quota Kaggle | risque principal |
|---|---|---|---|---|---|---|
| A1 KL vers `kaggle_sup` | 0 à +20 (protège) | faible | ½-1 j | ~3 h | 11-22 h | rien n'est appris / même dérive |
| A2 plancher d'entropie | ~0 | moyenne (sur le ~0) | ½ j | ~3 h | 11 h | réaplatissement |
| A3 part humaine 50 % | 0 à +10 (protège) | faible | 0 | ~3 h | 11-22 h | apprentissage ralenti |
| A4 acceptation contre l'ancre | 0 (protège) | élevée (sur la protection) | petit | 0 | inchangé | boucle figée |
| A5 800 simulations | ? | faible | 0-1 j | ~1 h | ×1,75 à ×4 par génération | quota pour un effet non démontré |
| A6 parties contre l'ancre | 0 à +20 | faible | 1-2 j | — | 11 h | complexité |
| A7 `reuse` 2 + G1 permanent | 0 (protège) | moyenne | 0 | ~3 h | 11 h | arrêt précoce |
| **B1 réseau plus gros** | **+30 à +100** | **moyenne** | ½-1 j | mesures ×2 à ×3,5 | 1,5-13 h | limite des données humaines |
| **B2 valeur sur Stockfish** | **+20 à +80** | **faible à moyenne** | 1 j | ~7 h (essai) | 0 (essai), ~4 h ensuite | gain nul (déjà vu en petit) |
| B3 données 2400+ | 0 à +50 | faible | ~0 | 2-6 h | 2-4 h | volume insuffisant |

## 5. Recommandation : voie B, en commençant par B2 sur le PC et B1 sur Kaggle en parallèle

1. **La voie A ne promet que de ne plus reculer.** Treize candidats, trois réglages de cibles et deux règles
   d'acceptation n'ont donné aucun signe de progrès au-delà de `kaggle_sup` (meilleurs : +9 biaisé, −5). L'analyse
   n'a pas trouvé de mécanisme simple à corriger (pas d'auto-renforcement de la confiance) mais une dérive lente et
   une non-transitivité, typiques d'un self-play trop petit pour son point de départ. Les corrections justifiées
   (A1, A4, A6, A7) sont protectrices ; le levier productif (A5) coûte 2 à 4 fois plus de quota sans preuve.
2. **La voie B attaque la faiblesse mesurée de `kaggle_sup` avec des gains mesurables sans ambiguïté** (v2,
   référence 2423 ± 26) et pour bien moins de quota : un essai B2 complet ne coûte que du temps de PC, un essai B1
   1,5 à 7 h de GPU.
3. **Ordre proposé** :
   - **PC : expérience minimale B2** (étiquetage de 1 M de positions, affinage, arène locale, puis v2 si ≥ +20) ;
     environ 7 h sur deux soirées, sur secteur, rien d'autre en charge.
   - **Kaggle, en parallèle : B1 à ~2 fois le calcul** (12 × 160, ou l'approfondissement 15 × 128 pour le premier
     essai), sur 48 M de positions (8 mois de plus extraits sur le PC), 92 000 pas : ~7 h de quota (1 h 30 pour
     l'approfondissement) ; arène locale puis v2 (~4 h la nuit).
   - **Si l'un des deux gagne** : un pré-entraînement combiné (réseau plus gros + valeur Stockfish, mélangée au
     résultat), mesuré en v2 ; il devient le nouveau réseau de référence et la nouvelle ancre.
   - **Le RL n'est pas abandonné** : le code de la boucle reste en état. S'il est repris, ce sera depuis le nouveau
     réseau, avec l'acceptation jugée contre l'ancre (A4), la KL vers le réseau de départ (A1) et le garde-fou G1
     permanent (A7), et seulement si l'on peut lui donner sensiblement plus de calcul par génération.
4. **Ce qui ferait changer d'avis** : si B2 et B1 donnent chacun moins de +20 Elo, la limite vient des données
   humaines ; la piste suivante serait alors la distillation des coups de Stockfish (variante de B2) ou un RL bien
   plus coûteux (A5 à 800 simulations sur le nouveau réseau), à décider à ce moment-là.
