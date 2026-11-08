"""B1 : données de pré-entraînement lichess_48m (16 mois) et archive du Dataset Kaggle.

  prepare_b1_data.py build  --raw data/lichess_48m_raw --ref data/lichess_24m --out data/lichess_48m
  prepare_b1_data.py zip    --data data/lichess_48m --out data/kaggle_upload/lichess_48m.zip
  prepare_b1_data.py verify --data data/lichess_48m [--zip data/kaggle_upload/lichess_48m.zip]

build : part de l'extraction brute (`prepare_data.py --months 2026-02 … 2026-09 2025-06 … 2026-01`, mêmes filtres
que lichess_24m) ; vérifie que les 8 premiers mois redonnent exactement lichess_24m (mêmes positions, mêmes
parties de validation) ; RETIRE de la partie entraînement toute position des lignes protégées (40 ouvertures v2 +
100 ouvertures de l'arène ancre, position initiale comprise, compteur des 50 coups ignoré :
`sf_value.opening_keys`, les mêmes exclusions que B2) ; recopie la validation telle quelle (elle contient celle de
lichess_24m, dont les 50 000 positions du critère) ; écrit `manifest.json` avec, pour chaque shard, positions,
taille et SHA-256 (le mode check du notebook les vérifie sur Kaggle).
zip : archive à plat (shards + manifest à la racine), comme make_kaggle_zip.py.
verify : relit le dossier (et l'archive) : empreintes, nombres de positions, aucune position protégée en entraînement.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = "lichess_48m"


from chesszero.dataset_check import check_shards, sha256_file  # noqa: E402


def concat(folder: Path, split: str, key: str) -> np.ndarray:
    from chesszero.data import shard_paths

    return np.concatenate([np.load(f)[key] for f in shard_paths(folder, split)])


def cmd_build(args) -> int:
    from chesszero.data import _write_shard, shard_paths
    from chesszero.sf_value import opening_keys, position_keys, protected_lines

    raw, ref, out = Path(args.raw), Path(args.ref), Path(args.out)
    mraw = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    mref = json.loads((ref / "manifest.json").read_text(encoding="utf-8"))
    n_ref_src = len(mref["sources"])
    if mraw["source"][:n_ref_src] != mref["source"] or len(mraw["source"]) != 16:
        raise SystemExit("l'extraction brute doit commencer par les 8 mois de lichess_24m, dans le même ordre (16 mois)")
    for a, b in zip(mraw["sources"][:n_ref_src], mref["sources"]):
        if (a["games_read"], a["games_kept"], a["positions"]) != (b["games_read"], b["games_kept"], b["positions"]):
            raise SystemExit(f"{a['source']} : extraction différente de lichess_24m ({a} contre {b})")
    t0 = time.perf_counter()
    for split in ("train", "val"):  # les positions de lichess_24m sont le début exact de l'extraction brute
        for key in ("boards", "moves", "results", "value_valid"):
            r, x = concat(ref, split, key), concat(raw, split, key)
            if not np.array_equal(x[:len(r)], r):
                raise SystemExit(f"{split}/{key} : les 8 premiers mois ne redonnent pas lichess_24m")
    print(f"8 premiers mois identiques à lichess_24m (entraînement et validation) ({time.perf_counter() - t0:.0f} s)")
    keys = opening_keys()
    lines = protected_lines()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} existe déjà (--overwrite pour le refaire)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shards, kept_total, removed_total = {}, 0, 0
    for f in shard_paths(raw, "train"):
        d = np.load(f)
        keep = np.array([k not in keys for k in position_keys(d["boards"])], bool)
        _write_shard(out / f.name, d["boards"][keep], d["moves"][keep], d["results"][keep], d["games"][keep],
                     d["value_valid"][keep])
        kept_total += int(keep.sum())
        removed_total += int((~keep).sum())
        print(f"\r{f.name} : {keep.sum():,} gardées, {(~keep).sum():,} retirées", end="", flush=True)
    print()
    for f in shard_paths(raw, "val"):
        shutil.copy2(f, out / f.name)
    for f in sorted(out.glob("*.npz")):
        n = len(np.load(f)["moves"])
        shards[f.name] = {"positions": n, "bytes": f.stat().st_size, "sha256": sha256_file(f)}
    n_val = sum(v["positions"] for k, v in shards.items() if k.startswith("val_"))
    manifest = {**mraw, "dataset": DATASET,
                "positions": {"train": kept_total, "val": n_val},
                "raw_positions": mraw["positions"],
                "exclusion": {"rule": "positions (hors compteur des 50 coups) des lignes protégées, position initiale "
                                      "comprise, retirées de la partie entraînement seulement (sf_value.opening_keys)",
                              "v2_lines": len(lines["v2"]), "anchor_lines": len(lines["anchor"]),
                              "protected_positions": len(keys), "train_removed": removed_total,
                              "train_removed_share": removed_total / (removed_total + kept_total)},
                "contains": {"lichess_24m_val": mref["positions"]["val"]},
                "shards": shards}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {out} : entraînement {kept_total:,} ({removed_total:,} retirées = {removed_total / (removed_total + kept_total):.1%}), "
          f"validation {n_val:,} ; {len(shards)} shards ({sum(v['bytes'] for v in shards.values()) / 1e9:.2f} Go)")
    return 0


def cmd_zip(args) -> int:
    data, out = Path(args.data), Path(args.out)
    files = sorted(data.glob("*.npz")) + [data / "manifest.json"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, f in enumerate(files, 1):
            z.write(f, f.name)
            print(f"\r{i}/{len(files)} fichiers", end="", flush=True)
    tmp.replace(out)
    print(f"\n{out} : {out.stat().st_size / 1e6:,.0f} Mo")
    return 0


def cmd_verify(args) -> int:
    from chesszero.sf_value import opening_keys

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    keys = opening_keys()
    t0 = time.perf_counter()
    errors = check_shards(lambda n: (data / n).read_bytes(), [p.name for p in data.glob("*.npz")],
                                       manifest, keys, "dossier")
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            bad = z.testzip()
            if bad:
                errors.append(f"archive : CRC faux pour {bad}")
            names = z.namelist()
            if any("/" in n for n in names):
                errors.append("archive : fichiers dans un sous-dossier (ils doivent être à la racine)")
            zm = json.loads(z.read("manifest.json"))
            if zm != manifest:
                errors.append("archive : manifest.json différent du dossier")
            errors += check_shards(z.read, names, manifest, None, "archive")
        size = Path(args.zip).stat().st_size
    print(f"{manifest['dataset']} : entraînement {manifest['positions']['train']:,}, validation "
          f"{manifest['positions']['val']:,}, {len(manifest['shards'])} shards ; vérifié en {time.perf_counter() - t0:.0f} s")
    if args.zip:
        print(f"archive {args.zip} : {size / 1e6:,.0f} Mo ({size:,} octets), {len(names)} fichiers")
    for e in errors:
        print("ERREUR :", e)
    print("COHÉRENT" if not errors else f"{len(errors)} ERREUR(S)")
    return 0 if not errors else 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from chesszero import cpu_guard

    cpu_guard.protect()  # EcoQoS désactivé sur ce processus et ses descendants (chesszero/cpu_guard.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--raw", default=str(ROOT / "data" / "lichess_48m_raw"))
    p.add_argument("--ref", default=str(ROOT / "data" / "lichess_24m"))
    p.add_argument("--out", default=str(ROOT / "data" / "lichess_48m"))
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("zip")
    p.add_argument("--data", default=str(ROOT / "data" / "lichess_48m"))
    p.add_argument("--out", default=str(ROOT / "data" / "kaggle_upload" / "lichess_48m.zip"))
    p = sub.add_parser("verify")
    p.add_argument("--data", default=str(ROOT / "data" / "lichess_48m"))
    p.add_argument("--zip", default=None)
    args = ap.parse_args()
    return {"build": cmd_build, "zip": cmd_zip, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
