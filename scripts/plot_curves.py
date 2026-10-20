"""Trace les courbes d'un entraînement à partir de son metrics.csv (image PNG).

  plot_curves.py checkpoints/tiny_sup                  # -> checkpoints/tiny_sup/curves.png
  plot_curves.py checkpoints/tiny_sup --out docs/tiny_sup_curves.png

Nécessite matplotlib (`pip install -e ".[plot]"`).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chesszero.train import read_metrics  # noqa: E402

# Palette de référence (clair) : bleu = entraînement, orange = validation.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
TRAIN, VAL = "#2a78d6", "#eb6834"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="dossier de l'entraînement (contient metrics.csv)")
    ap.add_argument("--out", help="image de sortie (défaut : <run>/curves.png)")
    args = ap.parse_args()

    rows = read_metrics(Path(args.run) / "metrics.csv")
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "val_full")}

    def series(split, key):
        return [r["step"] for r in by[split]], [r[key] for r in by[split]]

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)

    def line_pair(ax, key, title, ylabel):
        x, y = series("train", key)
        ax.plot(x, y, color=TRAIN, lw=2, label="entraînement (moyenne par fenêtre)")
        x, y = series("val", key)
        ax.plot(x, y, color=VAL, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label="validation (sous-ensemble fixe)")
        if by["val_full"]:
            x, y = series("val_full", key)
            ax.plot(x, y, ls="", marker="D", ms=8, mfc=SURFACE, mec=VAL, mew=2, label="validation complète (fin)")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("pas d'optimisation")

    line_pair(axes[0, 0], "policy_loss", "Perte politique (entropie croisée, coups légaux)", "nats")
    line_pair(axes[0, 1], "value_loss", "Perte valeur (MSE, positions « Normal » seulement)", "erreur quadratique moyenne")
    axes[0, 1].axhline(0.9, color=INK_2, lw=1, ls=":")
    axes[0, 1].annotate("prédire toujours 0 ≈ 0,9", (0.99, 0.9), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8.5, color=INK_2)

    ax = axes[1, 0]  # précisions : couleur = jeu de données, trait = top-1 (plein) / top-3 (tirets)
    for key, ls, name in (("top1", "-", "coup exact"), ("top3", "--", "dans le top 3")):
        x, y = series("train", key)
        ax.plot(x, [v * 100 for v in y], color=TRAIN, lw=2, ls=ls)
        x, y = series("val", key)
        ax.plot(x, [v * 100 for v in y], color=VAL, lw=2, ls=ls, marker="o", ms=6, mec=SURFACE, mew=1.5)
        ax.annotate(name, (x[-1], y[-1] * 100), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK_2, annotation_clip=False)
    ax.set_title("Précision du coup joué (parmi les coups légaux)", loc="left", fontsize=11, color=INK)
    ax.set_ylabel("%")
    ax.set_xlabel("pas d'optimisation")
    ax.margins(x=0.12)

    ax = axes[1, 1]
    x, y = series("train", "lr")
    ax.plot(x, y, color=INK_2, lw=2)
    ax.set_title("Taux d'apprentissage (montée puis cosinus)", loc="left", fontsize=11, color=INK)
    ax.set_ylabel("lr")
    ax.set_xlabel("pas d'optimisation")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9.5)
    last = by["val_full"][-1] if by["val_full"] else by["val"][-1]
    fig.suptitle(f"{Path(args.run).name} — pas {int(last['step'])} : top-1 {last['top1']:.1%}, top-3 {last['top3']:.1%}, "
                 f"perte {last['loss']:.3f}", x=0.01, ha="left", fontsize=12.5, color=INK)

    out = Path(args.out) if args.out else Path(args.run) / "curves.png"
    fig.savefig(out, dpi=130)
    print(f"Courbes écrites dans {out}")


if __name__ == "__main__":
    main()
