"""Ouvertures du protocole de mesure v2 (`match_stockfish.py`, `stockfish-ladder-v2`).

40 positions de départ, chacune jouée avec les deux couleurs : 32 lignes courantes (e4, d4, c4, Nf3) et
8 débuts rares (1.b4, 1.g4, 1.f4, 1.b3, 1.Nc3, 1...Nc6, 1...b6, Budapest). Critère de choix : écart de
moins d'un pion selon Stockfish 19 plein (8 threads, 2 s par position) ; l'Englund (+1,79) et le
Saint-Georges (+1,07) ont été écartés pour cette raison. Évaluation (centipions, point de vue des Blancs)
en commentaire. NE PAS MODIFIER : le protocole enregistré dans les résultats contient cette liste, et un
changement interdirait la comparaison avec les mesures déjà faites.
"""

OPENINGS_V2 = [
    # --- 1.e4 : ouvertes et semi-ouvertes
    ("Sicilienne Najdorf", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"),  # +28
    ("Sicilienne Alapin", "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4"),  # +12
    ("Sicilienne Rossolimo", "e2e4 c7c5 g1f3 b8c6 f1b5 g7g6"),  # +31
    ("Française Winawer", "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5"),  # +59
    ("Française avance", "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6"),  # +29
    ("Caro-Kann classique", "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5"),  # +37
    ("Espagnole Berlin", "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4"),  # +16
    ("Italienne Giuoco piano", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6"),  # +11
    ("Écossaise", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6"),  # +9
    ("Petrov", "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4"),  # +35
    ("Pirc autrichienne", "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7"),  # +57
    ("Alekhine moderne", "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3 c8g4"),  # +98
    ("Scandinave Qd6", "e2e4 d7d5 e4d5 d8d5 b1c3 d5d6"),  # +72
    ("Gambit du roi accepté", "e2e4 e7e5 f2f4 e5f4 g1f3 g7g5"),  # -40
    ("Viennoise", "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5"),  # -10
    # --- 1.d4
    ("Gambit dame refusé orthodoxe", "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7"),  # +10
    ("Gambit dame accepté", "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6 e2e3 e7e6"),  # +21
    ("Slave", "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4"),  # +25
    ("Semi-slave", "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7"),  # +12
    ("Nimzo-indienne", "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8"),  # 0
    ("Ouest-indienne", "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8a6"),  # +17
    ("Est-indienne classique", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8"),  # +55
    ("Grünfeld échange", "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3"),  # +29
    ("Benoni moderne", "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6"),  # +78
    ("Gambit Benko", "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 c4b5 a7a6"),  # +55
    ("Catalane", "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7"),  # +17
    ("Hollandaise Leningrad", "d2d4 f7f5 g2g3 g8f6 f1g2 g7g6 g1f3 f8g7"),  # +41
    ("Trompowsky", "d2d4 g8f6 c1g5 f6e4 g5f4 c7c5"),  # +38
    ("London", "d2d4 d7d5 c1f4 g8f6 e2e3 c7c5"),  # +11
    # --- débuts de flanc
    ("Anglaise symétrique", "c2c4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7"),  # +17
    ("Anglaise quatre cavaliers", "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6"),  # +9
    ("Réti", "g1f3 d7d5 c2c4 e7e6 g2g3 g8f6 f1g2 f8e7"),  # +15
    # --- débuts rares
    ("Polonaise 1.b4", "b2b4 e7e5 c1b2 f8b4 b2e5 g8f6"),  # -30
    ("Grob 1.g4", "g2g4 d7d5 f1g2 c7c6 h2h3 e7e5"),  # -99
    ("Bird 1.f4", "f2f4 d7d5 g1f3 g8f6 e2e3 g7g6"),  # -10
    ("Larsen 1.b3", "b2b3 e7e5 c1b2 b8c6 e2e3 d7d5"),  # -8
    ("Dunst 1.Nc3", "b1c3 d7d5 e2e4 d5d4 c3e2 e7e5"),  # -38
    ("Nimzowitsch 1...Nc6", "e2e4 b8c6 g1f3 d7d6 d2d4 g8f6"),  # +60
    ("Owen 1...b6", "e2e4 b7b6 d2d4 c8b7 f1d3 e7e6"),  # +93
    ("Budapest", "d2d4 g8f6 c2c4 e7e5 d4e5 f6g4"),  # +65
]
