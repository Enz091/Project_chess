"""Analyse du tampon de self-play d'une génération (parties rejouables, PGN, accord politique brute / cible MCTS).

Trois commandes, sur un dossier de shards (`<sortie>/selfplay/`), sans modifier le code de la boucle ni charger les
données Lichess :

  analyse_generation.py check    --buffer <dossier>                       les parties se rejouent-elles ? (coups, indices joués)
  analyse_generation.py pgn      --buffer <dossier> --out docs/analyse_gen0   PGN commentés : répétitions >= 40 points de
                                 matériel (5 avant le 60e demi-coup + 5 après), 5 mats les plus courts, 5 abandons (graine fixe)
  analyse_generation.py agreement --buffer <dossier> --checkpoint <réseau> [<autre réseau> ...] --out <json>
                                 accord entre la politique brute et la cible MCTS, par phase (ouverture / milieu / finale)

Les shards contiennent tout ce qu'il faut : coups UCI depuis la position initiale (ouverture du livre comprise),
résultat, raison de fin, Q racine de chaque position cherchée (les coups forcés ne sont pas enregistrés). Les PGN
donnent `Q(Blancs)` après chaque coup cherché et, au dernier coup, la raison de fin, le matériel restant et le dernier Q.

L'accord se calcule comme dans le rapport de session (chesszero/agreement.py) : biais du bruit de Dirichlet à la racine
rappelé dans le texte. Référence de la génération 0 (kaggle_sup, session de self-play seul) : voir docs/analyse_gen0/.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_shards(buffer: Path) -> list[tuple[str, dict]]:
    from chesszero.selfplay import load_shard

    return [(p.stem, load_shard(p)) for p in sorted(Path(buffer).glob("*.npz")) if p.with_suffix(".json").exists()]


def replay(shards, every: int = 10) -> dict:
    """Rejoue une partie sur `every` : chaque coup enregistré doit avoir l'indice `played`, et le nombre de demi-coups
    doit correspondre."""
    from chesszero.encoding import move_to_index

    games = positions = checked = bad = 0
    for _, d in shards:
        n = len(d["game_uid"])
        games += n
        positions += len(d["played"])
        for g in range(0, n, every):
            idx = np.nonzero(d["game"] == g)[0]
            recorded = {int(d["ply"][i]): i for i in idx}
            board = chess.Board()
            for ply, uci in enumerate(d["game_moves"][g].split()):
                if ply in recorded:
                    checked += 1
                    bad += move_to_index(chess.Move.from_uci(uci), board) != int(d["played"][recorded[ply]])
                board.push_uci(uci)
            bad += len(board.move_stack) != int(d["plies"][g])
    return {"games": games, "positions": positions, "positions_checked": checked, "inconsistencies": bad}


def cmd_check(args) -> int:
    r = replay(load_shards(args.buffer))
    print(f"{r['games']} parties, {r['positions']} positions ; {r['positions_checked']} positions rejouées ; "
          f"{r['inconsistencies']} incohérence(s)")
    return 1 if r["inconsistencies"] else 0


def cmd_pgn(args) -> int:
    from chesszero.selfplay import TERMINATIONS, material_points

    T = {name: i for i, name in enumerate(TERMINATIONS)}
    games = []
    for name, d in load_shards(args.buffer):
        for g in range(len(d["game_uid"])):
            board = chess.Board()
            for uci in d["game_moves"][g].split():
                board.push_uci(uci)
            games.append({"shard": name, "d": d, "g": g, "mat": material_points(board), "term": int(d["termination"][g]),
                          "plies": int(d["plies"][g]), "uid": str(d["game_uid"][g])})
    print(len(games), "parties")
    rep = [x for x in games if x["term"] == T["triple répétition"]]
    rep_mat = [x for x in rep if x["mat"] >= args.min_material]
    early = [x for x in rep_mat if x["plies"] < 60]
    late = [x for x in rep_mat if x["plies"] >= 60]
    print(f"répétitions {len(rep)} ; avec >= {args.min_material} points {len(rep_mat)} (avant le 60e demi-coup {len(early)}, "
          f"après {len(late)})")
    rng = random.Random(args.seed)
    sel_rep = sorted(rng.sample(early, min(5, len(early))), key=lambda x: x["plies"]) + \
        sorted(rng.sample(late, min(5, len(late))), key=lambda x: x["plies"])
    mates = sorted([x for x in games if x["term"] == T["mat"]], key=lambda x: (x["plies"], x["uid"]))[:5]
    resign = [x for x in games if x["term"] == T["abandon"]]
    sel_res = random.Random(args.seed).sample(resign, min(5, len(resign)))

    def q_white(d, i):  # camp au trait = Blancs aux demi-coups pairs (départ standard)
        q = float(d["root_q"][i])
        return q if int(d["ply"][i]) % 2 == 0 else -q

    def to_pgn(x, label):
        d, g = x["d"], x["g"]
        rec = {int(d["ply"][i]): i for i in np.nonzero(d["game"] == g)[0]}
        moves = d["game_moves"][g].split()
        rw = int(d["result_white"][g])
        game = chess.pgn.Game()
        game.headers.update({"Event": f"ChessZero self-play {args.name} ({label})".replace("  ", " "), "Site": "Kaggle",
                             "Date": args.date, "Round": "-", "White": args.player, "Black": args.player,
                             "Result": {1: "1-0", 0: "1/2-1/2", -1: "0-1"}[rw], "Termination": TERMINATIONS[x["term"]],
                             "PlyCount": str(x["plies"]), "GameUID": x["uid"], "Shard": x["shard"],
                             "BookIndex": str(int(d["start_index"][g])), "BookPlies": str(int(d["start_plies"][g])),
                             "Control": str(bool(d["control"][g])), "FinalMaterial": str(x["mat"])})
        last_q, node = None, game
        for ply, uci in enumerate(moves):
            node = node.add_variation(chess.Move.from_uci(uci))
            bits = []
            if ply == int(d["start_plies"][g]) - 1 and ply >= 0:
                bits.append("fin de l'ouverture du livre (coups suivants joués par le réseau)")
            if ply in rec:
                last_q = (ply, q_white(d, rec[ply]))
                bits.append(f"Q(Blancs)={last_q[1]:+.2f}")
            if ply == len(moves) - 1:
                side = "Blancs" if ply % 2 == 0 else "Noirs"
                tail = (f"FIN : {TERMINATIONS[x['term']]} après {len(moves)} demi-coups (dernier coup par les {side}) ; "
                        f"matériel restant {x['mat']} points")
                if last_q:
                    lag = len(moves) - 1 - last_q[0]
                    tail += (f" ; dernier Q enregistré = {last_q[1]:+.2f} (vue des Blancs) au demi-coup {last_q[0] + 1}"
                             + ("" if lag == 0 else f", {lag} demi-coup(s) avant la fin (coups forcés ou arbitrage non "
                                                    "enregistrés)"))
                if x["term"] == T["abandon"]:
                    tail += f" ; abandon des {'Noirs' if rw == 1 else 'Blancs'}"
                bits.append(tail)
            if bits:
                node.comment = " ; ".join(bits)
        return str(game)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, sel, label in (("repetitions_40pts.pgn", sel_rep, f"répétition, >= {args.min_material} points"),
                             ("mats_les_plus_courts.pgn", mates, "mat le plus court"),
                             ("abandons.pgn", sel_res, f"abandon, tirage graine {args.seed}")):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            for x in sel:
                f.write(to_pgn(x, label) + "\n\n")
        print(name, len(sel))
    lines = [f"# Analyse du tampon (source : {args.buffer})", "",
             "Fichiers générés par `scripts/analyse_generation.py pgn` (non versionnables à la légère : à regénérer).",
             "Coups en ordre depuis la position initiale (ouverture du livre comprise, `BookPlies` demi-coups) ; les commentaires",
             "donnent `Q(Blancs)` aux positions cherchées (les coups forcés ne sont pas enregistrés).", "",
             f"Répétitions : {len(rep)} ; avec >= {args.min_material} points de matériel : {len(rep_mat)} (avant le 60e "
             f"demi-coup : {len(early)} ; après : {len(late)}). Tirage graine {args.seed}, 5 + 5.", ""]
    for title, sel in (("repetitions_40pts.pgn", sel_rep), ("mats_les_plus_courts.pgn", mates), ("abandons.pgn", sel_res)):
        lines += [f"## {title}", "", "| # | UID | demi-coups | fin | matériel | résultat | départ livre |",
                  "|---|---|---|---|---|---|---|"]
        for k, x in enumerate(sel, 1):
            d, g = x["d"], x["g"]
            lines.append(f"| {k} | {x['uid']} | {x['plies']} | {TERMINATIONS[x['term']]} | {x['mat']} | "
                         f"{int(d['result_white'][g])} | {int(d['start_plies'][g])} |")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return 0


def cmd_agreement(args) -> int:
    import torch

    from chesszero.agreement import (DIRICHLET_NOTE, PHASE_LABELS, PHASES, gather, policy_agreement, sample_positions,
                                     summarize)
    from chesszero.checkpoint import load_for_inference
    from chesszero.device import get_device

    shards = [d for _, d in load_shards(args.buffer)]
    picks = sample_positions(shards, args.positions, args.seed, exclude_heldout=None if args.all_positions else
                             args.heldout_fraction)
    batch = gather(shards, picks)
    device = get_device(args.device)
    torch.set_num_threads(args.threads)
    result = {"buffer": str(args.buffer), "positions": len(picks), "seed": args.seed,
              "sample": "toutes les positions" if args.all_positions else
              f"positions d'entraînement (parties mises de côté exclues, fraction {args.heldout_fraction})",
              "definitions": "accord = coup préféré du réseau (softmax sur les coups légaux) = coup le plus visité ; "
                             "KL = KL(cible MCTS || politique brute)", "note": DIRICHLET_NOTE, "networks": {}}
    for path in args.checkpoint:
        model, meta = load_for_inference(path, device)
        agree, kl = policy_agreement(model, batch, device)
        s = summarize(batch, agree, kl)
        result["networks"][str(path)] = {"step": meta.get("step"), "phases": s}
        print(f"\n{path} (pas {meta.get('step')}) — {len(picks)} positions")
        for ph in ("all",) + PHASES:
            c = s[ph]
            if c["n"]:
                print(f"  {PHASE_LABELS[ph]:<10} n={c['n']:>6}  accord {c['agree']:.1%}  KL {c['kl']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("->", args.out)
    return 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("check", "pgn", "agreement"):
        p = sub.add_parser(name)
        p.add_argument("--buffer", required=True, type=Path, help="dossier de shards (<sortie>/selfplay)")
        if name == "pgn":
            p.add_argument("--out", required=True)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--min-material", type=int, default=40, help="matériel minimal des répétitions (points)")
            p.add_argument("--name", default="", help="nom de la génération dans l'en-tête Event (ex. « génération 0 »)")
            p.add_argument("--date", default="????.??.??", help="en-tête Date (AAAA.MM.JJ)")
            p.add_argument("--player", default="réseau (200 sim.)", help="noms des joueurs dans les en-têtes")
        if name == "agreement":
            p.add_argument("--checkpoint", nargs="+", required=True, help="un ou plusieurs réseaux (même échantillon)")
            p.add_argument("--positions", type=int, default=20000)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--all-positions", action="store_true",
                           help="toutes les positions (défaut : positions d'entraînement, parties mises de côté exclues)")
            p.add_argument("--heldout-fraction", type=float, default=0.10)
            p.add_argument("--device", default="cpu")
            p.add_argument("--threads", type=int, default=8)
            p.add_argument("--out", default=None, help="fichier JSON des résultats")
    args = ap.parse_args()
    sys.exit({"check": cmd_check, "pgn": cmd_pgn, "agreement": cmd_agreement}[args.command](args))


if __name__ == "__main__":
    main()
