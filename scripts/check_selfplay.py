"""Vérifie les fichiers de self-play d'un dossier et résume les parties.

  check_selfplay.py selfplay_essai            # tous les shards du dossier
  check_selfplay.py selfplay_essai --pgn parties.pgn --book books/lichess_2000.tsv.gz

Vérifications (chesszero.selfplay.verify_shard, chaque partie rejouée depuis la position initiale) : positions
enregistrées = positions de la partie, coups légaux enregistrés = ceux de la position décodée, coup joué présent
et visité, visites >= simulations, signe du résultat pour les deux camps, fin de partie cohérente (mat, nulles,
tables, abandon, limite), demi-coups d'abandon des parties de contrôle. Puis résumé : longueurs, fins,
résultats, départs, diversité des ouvertures, faux abandons des parties de contrôle.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--config", default="kaggle")
    ap.add_argument("--syzygy", default=str(ROOT / "tools" / "syzygy"), help="'' = ne pas revérifier les arbitrages")
    ap.add_argument("--book", default=str(ROOT / "books" / "lichess_2000.tsv.gz"))
    ap.add_argument("--pgn", default=None, help="écrire les parties en PGN")
    args = ap.parse_args()

    from chesszero.book import OpeningBook
    from chesszero.selfplay import (TERMINATIONS, T_RESIGN, SelfPlayConfig, describe_repetitions, list_shards,
                                    load_shard, repetition_stats, verify_shard)
    from chesszero.syzygy import Tablebase

    cfg = SelfPlayConfig.named(args.config)
    tb = Tablebase.open(args.syzygy, wdl_only=True) if args.syzygy else None
    book = OpeningBook.load(args.book) if args.book and Path(args.book).exists() else None
    shards = sorted(list_shards(args.dirs), key=lambda s: s["name"])
    games, n_positions, errors, loaded = [], 0, [], []
    for s in shards:
        data = load_shard(s["npz"])
        loaded.append(data)
        errs = verify_shard(data, cfg, tb)
        errors += [f"{s['name']} : {e}" for e in errs]
        n_positions += len(data["played"])
        for g in range(len(data["game_uid"])):
            rows = np.nonzero(data["game"] == g)[0]
            games.append({k: data[k][g] for k in ("game_uid", "game_moves", "start_index", "start_plies",
                                                   "result_white", "termination", "plies", "control", "would_resign")}
                         | {"positions": len(rows), "visits": [int(data["policy_visits"][data["policy_offsets"][r]:
                                                                                         data["policy_offsets"][r + 1]].sum())
                                                               for r in rows]})
        if book is not None:  # le début de chaque partie partie du livre = la ligne du livre
            for g in range(len(data["game_uid"])):
                i = int(data["start_index"][g])
                if i >= 0 and not str(data["game_moves"][g]).startswith(book.lines[i]):
                    errors.append(f"{s['name']} : partie {g} : début différent de la ligne {i} du livre")

    print(f"{len(shards)} shards, {len(games)} parties, {n_positions} positions enregistrées")
    print(f"Vérifications : {'AUCUNE ERREUR' if not errors else f'{len(errors)} ERREUR(S)'}")
    for e in errors[:30]:
        print("  " + e)
    if not games:
        return

    plies = [int(g["plies"]) for g in games]
    played = [int(g["plies"]) - int(g["start_plies"]) for g in games]
    print(f"\nLongueur (demi-coups depuis la position initiale) : moyenne {statistics.mean(plies):.0f}, "
          f"min {min(plies)}, max {max(plies)} ; joués en self-play : moyenne {statistics.mean(played):.0f}")
    print("Fins : " + ", ".join(f"{TERMINATIONS[t]} {n}" for t, n in sorted(Counter(int(g['termination']) for g in games).items())))
    res = Counter(int(g["result_white"]) for g in games)
    print(f"Résultats : Blancs gagnent {res[1]}, nulles {res[0]}, Noirs gagnent {res[-1]}")
    print(describe_repetitions(repetition_stats(loaded)))
    visits = [v for g in games for v in g["visits"]]
    print(f"Visites à la racine (réutilisation de l'arbre comprise) : moyenne {statistics.mean(visits):.0f}, "
          f"min {min(visits)}, max {max(visits)} (simulations par coup : {cfg.simulations})")

    from_book = [g for g in games if int(g["start_index"]) >= 0]
    print(f"\nDéparts : {len(from_book)} du livre, {len(games) - len(from_book)} de la position initiale ; "
          f"positions de départ distinctes : {len({int(g['start_index']) for g in from_book})} sur {len(from_book)}")
    firsts = Counter()
    for g in games:
        b, sans = chess.Board(), []
        for u in str(g["game_moves"]).split()[:4]:
            m = chess.Move.from_uci(u)
            sans.append(b.san(m))
            b.push(m)
        firsts[" ".join(sans)] += 1
    print("Quatre premiers demi-coups : " + " | ".join(f"{k} ({n})" for k, n in firsts.most_common(12)))
    at20 = Counter()
    for g in games:
        moves = str(g["game_moves"]).split()
        if len(moves) >= 30:
            b = chess.Board()
            for u in moves[:30]:
                b.push_uci(u)
            at20[b.epd()] += 1
    print(f"Positions distinctes au demi-coup 30 : {len(at20)} sur {sum(at20.values())} parties assez longues")

    controls = [g for g in games if bool(g["control"])]
    triggered = [(g, c) for g in controls for c in (0, 1) if int(g["would_resign"][c]) >= 0]
    false = [1 for g, c in triggered if int(g["result_white"]) != (-1 if c == 0 else 1)]
    print(f"\nParties de contrôle : {len(controls)} ; abandon qui se serait déclenché : {len(triggered)} ; "
          f"faux abandons (le camp n'a pas perdu) : {len(false)}")
    resigned = [g for g in games if int(g["termination"]) == T_RESIGN]
    print(f"Parties finies par abandon : {len(resigned)}")

    if args.pgn:
        with open(args.pgn, "w", encoding="utf-8", newline="\n") as f:
            for g in games:
                pgn = chess.pgn.Game()
                pgn.headers.update({"Event": "ChessZero self-play", "Site": "-", "Round": str(g["game_uid"]),
                                    "White": "ChessZero", "Black": "ChessZero",
                                    "Result": {1: "1-0", 0: "1/2-1/2", -1: "0-1"}[int(g["result_white"])],
                                    "Termination": TERMINATIONS[int(g["termination"])],
                                    "Annotator": f"départ {'livre' if int(g['start_index']) >= 0 else 'initial'} "
                                                 f"(demi-coup {int(g['start_plies'])})"})
                node = pgn
                for u in str(g["game_moves"]).split():
                    node = node.add_variation(chess.Move.from_uci(u))
                print(pgn, file=f, end="\n\n")
        print(f"PGN : {args.pgn}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
