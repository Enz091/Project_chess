"""Télécharge les tables Syzygy 3-4-5 pièces depuis le miroir officiel de Lichess, en vérifiant chaque fichier
contre les sommes SHA-256 publiées à côté.

  download_syzygy.py --out /tmp/syzygy --wdl-only     # Kaggle : WDL seules (145 fichiers, 396 Mo), assez pour
                                                      # l'arbitrage du self-play (voir chesszero/syzygy.py)
  download_syzygy.py --out tools/syzygy               # WDL + DTZ (290 fichiers, 984 Mo) : choix du coup en finale

Les fichiers déjà présents et conformes ne sont pas retéléchargés. Un fichier non conforme est supprimé et retéléchargé
(3 essais). Code de sortie 1 si un fichier reste absent ou non conforme.
"""

import argparse
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://tablebase.lichess.ovh/tables/standard"
UA = {"User-Agent": "chesszero-syzygy/0.1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def official_list(wdl_only: bool, max_pieces: int = 5) -> list[tuple[str, str]]:
    """(nom de fichier, SHA-256) des tables voulues, d'après le fichier `sha256` publié par le miroir."""
    with urllib.request.urlopen(urllib.request.Request(f"{BASE}/sha256", headers=UA), timeout=60) as r:
        text = r.read().decode()
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        name = parts[1].rsplit("/", 1)[-1]
        stem, _, ext = name.partition(".")
        if ext not in ("rtbw", "rtbz") or len(stem) - 1 > max_pieces or (wdl_only and ext != "rtbw"):
            continue
        out.append((name, parts[0]))
    return sorted(set(out))


def fetch(name: str, digest: str, out: Path, tries: int = 3) -> str | None:
    """Télécharge un fichier et vérifie son SHA-256 ; renvoie None si tout va bien, sinon le message d'erreur."""
    path = out / name
    if path.exists() and sha256(path) == digest:
        return None
    sub = "3-4-5-wdl" if name.endswith(".rtbw") else "3-4-5-dtz"
    err = "?"
    for attempt in range(tries):
        try:
            tmp = path.with_name(name + ".part")
            with urllib.request.urlopen(urllib.request.Request(f"{BASE}/{sub}/{name}", headers=UA), timeout=120) as r, \
                    open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            if sha256(tmp) == digest:
                tmp.replace(path)
                return None
            tmp.unlink()
            err = "SHA-256 différent"
        except OSError as exc:
            err = repr(exc)
        time.sleep(2 * (attempt + 1))
    return f"{name} : {err}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wdl-only", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = official_list(args.wdl_only)
    start = time.time()
    with ThreadPoolExecutor(args.threads) as pool:
        errors = [e for e in pool.map(lambda f: fetch(*f, out), files) if e]
    size = sum((out / n).stat().st_size for n, _ in files if (out / n).exists())
    print(f"{len(files) - len(errors)}/{len(files)} fichiers conformes (SHA-256 officiel), {size / 1e6:.1f} Mo, "
          f"{time.time() - start:.0f} s -> {out}")
    for e in errors:
        print("  ERREUR", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
