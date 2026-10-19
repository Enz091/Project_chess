"""Réseau à deux têtes façon AlphaZero : tronc ResNet, tête politique, tête valeur.

Entrée  : (B, N_PLANES, 8, 8), tenseur de `encoding.encode_board` (point de vue du camp au trait).
Sorties : - logits de politique (B, N_ACTIONS), un par coup possible de l'espace 8x8x73 ;
            les coups illégaux ne sont PAS masqués ici (voir `masked_log_softmax`) ;
          - valeur (B,) dans [-1, 1], du point de vue du camp au trait (+1 = je gagne).
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from chesszero.encoding import N_ACTIONS, N_MOVE_PLANES, N_PLANES

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_MASK_VALUE = -1e9  # logit des coups illégaux (fini : évite 0 * -inf = nan dans la perte)


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class ModelConfig:
    n_blocks: int = 4  # nombre de blocs résiduels
    n_filters: int = 64  # largeur du tronc (canaux)
    value_hidden: int = 64  # taille de la couche cachée de la tête valeur

    @classmethod
    def from_toml(cls, path: str | Path) -> ModelConfig:
        with open(path, "rb") as f:
            return cls(**tomllib.load(f)["model"])

    @classmethod
    def named(cls, name: str) -> ModelConfig:
        """Charge configs/<name>.toml (« tiny », « kaggle »)."""
        return cls.from_toml(CONFIG_DIR / f"{name}.toml")

    def to_dict(self) -> dict:
        """Pour stocker dans le champ `config` d'un checkpoint."""
        return asdict(self)


# --------------------------------------------------------------------------- réseau


class ResBlock(nn.Module):
    """x -> relu(x + BN(conv(relu(BN(conv(x))))))."""

    def __init__(self, n_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class ChessNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        f = config.n_filters

        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(f) for _ in range(config.n_blocks)])

        # Tête politique : un score par (case de départ, plan de coup).
        self.policy_head = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(),
            nn.Conv2d(f, N_MOVE_PLANES, 1),
        )
        # Tête valeur : résumé de l'échiquier -> un scalaire.
        self.value_conv = nn.Sequential(nn.Conv2d(f, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
        self.value_fc = nn.Sequential(
            nn.Linear(64, config.value_hidden), nn.ReLU(), nn.Linear(config.value_hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))

        # (B, 73, 8, 8) -> (B, 8, 8, 73) -> (B, 4672) : indice = rangée*8*73 + colonne*73 + plan
        # = case_départ * 73 + plan, la convention de encoding.move_to_index.
        logits = self.policy_head(h).permute(0, 2, 3, 1).reshape(x.shape[0], N_ACTIONS)

        value = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(1)
        return logits, value


def build_model(config: ModelConfig | str) -> ChessNet:
    """`config` : un ModelConfig ou le nom d'un fichier de configs/ (« tiny », « kaggle »)."""
    if isinstance(config, str):
        config = ModelConfig.named(config)
    return ChessNet(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- politique et perte


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """log-probabilités sur les coups légaux seulement (les illégaux valent ~ -1e9).

    `legal_mask` : booléen (B, N_ACTIONS). Calcul en float32 (stable même sous autocast fp16).
    """
    return F.log_softmax(logits.float().masked_fill(~legal_mask, _MASK_VALUE), dim=-1)


def masked_policy(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Probabilités des coups : somment à 1 sur les coups légaux, 0 ailleurs."""
    return masked_log_softmax(logits, legal_mask).exp()


def policy_value_loss(
    logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Perte = entropie croisée (politique, coups illégaux masqués) + erreur quadratique (valeur).

    `target_policy` : soit des indices de coups (B,) entiers (pré-entraînement : le coup joué),
    soit une distribution (B, N_ACTIONS) (self-play : visites MCTS normalisées).
    `target_value` : (B,) dans [-1, 1].
    Renvoie (perte totale, {"policy": ..., "value": ...}) ; les deux termes ont le même poids.
    """
    log_p = masked_log_softmax(logits, legal_mask)
    if target_policy.dim() == 1:
        policy_loss = F.nll_loss(log_p, target_policy)
    else:
        policy_loss = -(target_policy * log_p).sum(dim=-1).mean()
    value_loss = F.mse_loss(value.float(), target_value.float())
    return policy_loss + value_loss, {"policy": policy_loss.item(), "value": value_loss.item()}
