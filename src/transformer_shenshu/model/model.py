"""Hybrid Encoder → 6 bar-id classification heads (Shenshu-42M-v1).

Primary: structured features → Transformer encoder → 6 Linear heads over LUT (~12k).
Aux / routing: dedicated 六亲 MLP → 8-way true_ke; ke embedding is teacher-forced into
bar trunk during train so heads disambiguate within the 八刻 family.
Optional VQ bottleneck / LM decoder modules are disabled by default (use_vq / use_lm).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 0
    dim_feedforward: int = 512
    dropout: float = 0.1
    codebook_size: int = 1024
    code_length: int = 16
    vocab_size: int = 8192
    max_target_len: int = 512
    n_sex: int = 3
    n_gan: int = 11
    n_zhi: int = 13
    n_zodiac: int = 14
    n_count: int = 16
    n_input_slots: int = 16
    n_bars: int = 12000
    n_sections: int = 6
    use_vq: bool = False
    use_lm: bool = False
    aux_true_ke: bool = True
    aux_true_ke_weight: float = 2.0


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = z_e.reshape(-1, self.dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q = self.embedding(indices).view_as(z_e)
        loss_codebook = F.mse_loss(z_q.detach(), z_e)
        loss_commit = F.mse_loss(z_q, z_e.detach())
        vq_loss = loss_codebook + self.beta * loss_commit
        z_q_st = z_e + (z_q - z_e).detach()
        indices = indices.view(z_e.shape[0], z_e.shape[1])
        return z_q_st, indices, vq_loss


def true_ke_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Classical pick_ke with 0-based zodiac indices."""
    father = (batch["father_zodiac"] - 1).clamp(min=0)
    mother = (batch["mother_zodiac"] - 1).clamp(min=0)
    spouse = (batch["spouse_zodiac"] - 1).clamp(min=0)
    brothers = batch["brothers"]
    return (father + 3 * mother + 5 * brothers + 7 * spouse) % 8


class ShenShuModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        self.sex_emb = nn.Embedding(c.n_sex, c.d_model)
        self.gan_emb = nn.Embedding(c.n_gan, c.d_model)
        self.zhi_emb = nn.Embedding(c.n_zhi, c.d_model)
        self.zodiac_emb = nn.Embedding(c.n_zodiac, c.d_model)
        self.count_emb = nn.Embedding(c.n_count, c.d_model)
        self.slot_emb = nn.Embedding(c.n_input_slots, c.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.nhead,
            dim_feedforward=c.dim_feedforward,
            dropout=c.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=c.num_encoder_layers)

        self.section_emb = nn.Embedding(c.n_sections, c.d_model)
        self.ke_emb = nn.Embedding(8, c.d_model)
        self.bar_trunk = nn.Sequential(
            nn.Linear(c.d_model, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, c.d_model),
            nn.GELU(),
        )
        self.bar_heads = nn.ModuleList(
            [nn.Linear(c.d_model, c.n_bars) for _ in range(c.n_sections)]
        )

        # Dedicated 六亲 → true_ke (does not go through full encoder)
        # slots: father=9, mother=10, brothers=11, spouse=12 in encode_features order
        self.qin_true_ke = nn.Sequential(
            nn.Linear(c.d_model * 4, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, c.d_model),
            nn.GELU(),
            nn.Linear(c.d_model, 8),
        )

        self.to_codes = nn.Linear(c.d_model, c.code_length * c.d_model) if c.use_vq else None
        self.vq = VectorQuantizer(c.codebook_size, c.d_model) if c.use_vq else None

        if c.use_lm and c.num_decoder_layers > 0:
            self.tgt_emb = nn.Embedding(c.vocab_size, c.d_model)
            self.tgt_pos = nn.Embedding(c.max_target_len, c.d_model)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=c.d_model,
                nhead=c.nhead,
                dim_feedforward=c.dim_feedforward,
                dropout=c.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=c.num_decoder_layers)
            self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        else:
            self.tgt_emb = None
            self.tgt_pos = None
            self.decoder = None
            self.lm_head = None

    def encode_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.sex_emb(batch["sex"]),
            self.gan_emb(batch["year_gan"]),
            self.zhi_emb(batch["year_zhi"]),
            self.gan_emb(batch["month_gan"]),
            self.zhi_emb(batch["month_zhi"]),
            self.gan_emb(batch["day_gan"]),
            self.zhi_emb(batch["day_zhi"]),
            self.gan_emb(batch["hour_gan"]),
            self.zhi_emb(batch["hour_zhi"]),
            self.zodiac_emb(batch["father_zodiac"]),
            self.zodiac_emb(batch["mother_zodiac"]),
            self.count_emb(batch["brothers"]),
            self.zodiac_emb(batch["spouse_zodiac"]),
            self.count_emb(batch["sons"]),
            self.count_emb(batch["daughters"]),
        ]
        x = torch.stack(parts, dim=1)
        b, s, d = x.shape
        if s < self.cfg.n_input_slots:
            pad = torch.zeros(b, self.cfg.n_input_slots - s, d, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, : self.cfg.n_input_slots]
        slots = torch.arange(self.cfg.n_input_slots, device=x.device).unsqueeze(0).expand(b, -1)
        return x + self.slot_emb(slots)

    def qin_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Raw 六亲 embeddings (no slot/encoder) for dedicated true_ke MLP."""
        return torch.cat(
            [
                self.zodiac_emb(batch["father_zodiac"]),
                self.zodiac_emb(batch["mother_zodiac"]),
                self.count_emb(batch["brothers"]),
                self.zodiac_emb(batch["spouse_zodiac"]),
            ],
            dim=-1,
        )

    def encode_hidden(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(self.encode_features(batch))

    def bar_logits_from_hidden(
        self, h: torch.Tensor, ke_ids: torch.Tensor
    ) -> torch.Tensor:
        """[B,6,n_bars]; ke_ids [B] indexes ke_emb (teacher-forced or predicted)."""
        pooled = h.mean(dim=1)
        ke = self.ke_emb(ke_ids.long())
        logits = []
        for i, head in enumerate(self.bar_heads):
            feat = self.bar_trunk(pooled + self.section_emb.weight[i] + ke)
            logits.append(head(feat))
        return torch.stack(logits, dim=1)

    def encode(self, batch: dict[str, torch.Tensor]):
        h = self.encode_hidden(batch)
        pooled = h.mean(dim=1)
        if self.vq is not None and self.to_codes is not None:
            z_e = self.to_codes(pooled).view(-1, self.cfg.code_length, self.cfg.d_model)
            return self.vq(z_e)
        b = pooled.size(0)
        z_q = pooled.unsqueeze(1).expand(-1, self.cfg.code_length, -1)
        indices = torch.zeros(b, self.cfg.code_length, dtype=torch.long, device=pooled.device)
        vq_loss = torch.zeros((), device=pooled.device)
        return z_q, indices, vq_loss

    def decode(self, memory: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        if self.decoder is None or self.lm_head is None:
            raise RuntimeError("LM decoder disabled (use_lm=False).")
        b, t = tgt_ids.shape
        pos = torch.arange(t, device=tgt_ids.device).unsqueeze(0).expand(b, -1)
        tok = self.tgt_emb(tgt_ids) + self.tgt_pos(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=tgt_ids.device)
        out = self.decoder(tok, memory, tgt_mask=causal)
        return self.lm_head(out)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.encode_hidden(batch)
        pooled = h.mean(dim=1)

        ke_tgt = true_ke_from_batch(batch)
        ke_logits = self.qin_true_ke(self.qin_features(batch))
        out: dict[str, torch.Tensor] = {
            "true_ke_logits": ke_logits,
            "true_ke_target": ke_tgt,
        }
        loss_ke = F.cross_entropy(ke_logits, ke_tgt.long())
        out["loss_true_ke"] = loss_ke
        out["true_ke_acc"] = (ke_logits.argmax(-1) == ke_tgt).float().mean()

        # Teacher force gold ke into bar heads while training; use pred at eval
        if self.training:
            ke_for_bar = ke_tgt
        else:
            ke_for_bar = ke_logits.argmax(dim=-1)

        bar_logits = self.bar_logits_from_hidden(h, ke_for_bar)
        out["bar_logits"] = bar_logits

        loss = self.cfg.aux_true_ke_weight * loss_ke
        if "verse_ids" in batch:
            B, S, C = bar_logits.shape
            loss_bar = F.cross_entropy(
                bar_logits.reshape(B * S, C),
                batch["verse_ids"].reshape(B * S).long(),
            )
            out["loss_bar"] = loss_bar
            loss = loss + loss_bar
            preds = bar_logits.argmax(dim=-1)
            out["bar_acc"] = (preds == batch["verse_ids"]).float().mean()
            out["bar_acc_per_sec"] = (preds == batch["verse_ids"]).float().mean(dim=0)

        out["loss"] = loss
        out["vq_loss"] = torch.zeros((), device=h.device)

        if self.cfg.use_vq and self.vq is not None:
            z_e = self.to_codes(pooled).view(-1, self.cfg.code_length, self.cfg.d_model)
            z_q, indices, vq_loss = self.vq(z_e)
            out["vq_loss"] = vq_loss
            out["code_indices"] = indices

        if self.cfg.use_lm and self.decoder is not None and "tgt_in" in batch:
            logits = self.decode(h, batch["tgt_in"])
            out["logits"] = logits
            if "tgt_out" in batch:
                out["loss_lm"] = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch["tgt_out"].reshape(-1),
                    ignore_index=0,
                )
                out["loss"] = out["loss"] + out["loss_lm"]

        return out

    def predict_verse_ids(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            out = self.forward(batch)
            return out["bar_logits"].argmax(dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
