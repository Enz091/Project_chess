"""Empaquette un dossier de shards (data/<nom>) en une archive .zip à téléverser comme Dataset Kaggle.

Les fichiers sont à la racine de l'archive : Kaggle décompresse le .zip à la création du dataset et
les shards apparaissent directement dans /kaggle/input/<nom-du-dataset>/ (train_00000.npz, ..., manifest.json).
Les shards se compressent très bien (~7x) : 1,8 Go de données -> ~260 Mo à envoyer.

Exemple : make_kaggle_zip.py data/lichess_24m --out data/kaggle/lichess_24m.zip
"""

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*.npz")) + [Path(args.data_dir) / "manifest.json"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    raw = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, f in enumerate(files, 1):
            z.write(f, f.name)
            raw += f.stat().st_size
            print(f"\r{i}/{len(files)} fichiers", end="", flush=True)
    tmp.replace(out)
    print(f"\n{out} : {out.stat().st_size / 1e6:,.0f} Mo (données brutes : {raw / 1e6:,.0f} Mo)")


if __name__ == "__main__":
    main()
