"""Archive d'une sortie de session de boucle, à téléverser comme Dataset Kaggle privé.

  make_kaggle_output_zip.py data/kaggle_out/loop_session2 --out data/kaggle_upload/loop_session2.zip

Utile quand Kaggle ne propose pas la sortie d'une version (version « failed ») : le Dataset est reconnu par son contenu
(`selfplay/`, `sp_train/`, `loop/`), à n'importe quelle profondeur sous /kaggle/input. Les trois dossiers sont placés
directement à la RACINE de l'archive (pas de dossier intermédiaire). Vérifie que la sortie est cohérente avant d'écrire.
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("loop", "selfplay", "sp_train")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("output", type=Path, help="dossier de la sortie téléchargée (loop/, selfplay/, sp_train/)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from chesszero.loop import classify_output, last_session, read_loop_state

    info = classify_output(args.output)
    if info["kind"] != "loop":
        sys.exit(f"{args.output} n'est pas une sortie de boucle cohérente : {info}")
    session = last_session(read_loop_state(args.output / "loop" / "loop_state.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".zip.tmp")
    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for folder in FOLDERS:
            for p in sorted((args.output / folder).rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(args.output).as_posix())
                    count += 1
    tmp.replace(args.out)
    with zipfile.ZipFile(args.out) as z:
        bad = z.testzip()
        names = z.namelist()
    roots = sorted({n.split("/")[0] for n in names})
    print(f"{args.out} : {count} fichiers, {args.out.stat().st_size / 1e6:.1f} Mo ; session de boucle n°{session}, "
          f"{info['shards']} shards, {info['positions']:,} positions ; racine de l'archive : {roots} ; "
          f"test de l'archive : {'OK' if bad is None else 'FICHIER CORROMPU ' + bad}")
    if bad is not None or roots != sorted(FOLDERS):
        sys.exit(1)


if __name__ == "__main__":
    main()
