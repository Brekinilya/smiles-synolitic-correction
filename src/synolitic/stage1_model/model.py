"""Small encoder-decoder Transformer for the sequence reversal task (Setup 1).

Architecture per the proposal: 2 encoder + 2 decoder layers, 4 attention
heads, d_model=64, d_ff=256, ~236K parameters. Implemented with explicit
blocks instead of ``nn.Transformer`` so that per-head attention matrices and
per-layer hidden states can be extracted exactly. Pre-LN residual layout is
used for stable training without a warmup schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from synolitic.common.schemas import BOS, EOS, PAD, VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 64


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


def _feed_forward(d_model: int, d_ff: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
    )


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = _feed_forward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, key_padding_mask: Tensor, need_attn: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        h = self.ln1(x)
        a, w = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=need_attn,
            average_attn_weights=False,
        )
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, w


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = _feed_forward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        y: Tensor,
        memory: Tensor,
        causal_mask: Tensor,
        tgt_key_padding_mask: Tensor,
        mem_key_padding_mask: Tensor,
        need_attn: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        h = self.ln1(y)
        a, self_w = self.self_attn(
            h, h, h,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=need_attn,
            average_attn_weights=False,
        )
        y = y + self.drop(a)
        h = self.ln2(y)
        a, cross_w = self.cross_attn(
            h, memory, memory,
            key_padding_mask=mem_key_padding_mask,
            need_weights=need_attn,
            average_attn_weights=False,
        )
        y = y + self.drop(a)
        y = y + self.drop(self.ff(self.ln3(y)))
        return y, self_w, cross_w


class ReversalTransformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.scale = math.sqrt(c.d_model)
        self.embed = nn.Embedding(c.vocab_size, c.d_model, padding_idx=PAD)
        self.pos = SinusoidalPositionalEncoding(c.d_model, c.max_len)
        self.drop = nn.Dropout(c.dropout)
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(c.d_model, c.n_heads, c.d_ff, c.dropout) for _ in range(c.n_layers)
        )
        self.enc_norm = nn.LayerNorm(c.d_model)
        self.dec_norm = nn.LayerNorm(c.d_model)
        self.out_proj = nn.Linear(c.d_model, c.vocab_size)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(
        self, src: Tensor, need_attn: bool = False
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
        """Returns (memory, src_key_padding_mask, per-layer hiddens, per-layer attn).

        ``memory`` is the final-LN encoder output; ``hiddens`` are raw
        per-layer outputs (before the final LN). Attention weights are
        [B, heads, L, L] per layer when ``need_attn``.
        """
        src_kpm = src.eq(PAD)
        x = self.drop(self.pos(self.embed(src) * self.scale))
        hiddens: list[Tensor] = []
        attns: list[Tensor] = []
        for layer in self.encoder_layers:
            x, w = layer(x, src_kpm, need_attn)
            hiddens.append(x)
            if need_attn:
                attns.append(w)
        return self.enc_norm(x), src_kpm, hiddens, attns

    def decode(
        self,
        tgt_in: Tensor,
        memory: Tensor,
        src_kpm: Tensor,
        need_attn: bool = False,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        t = tgt_in.size(1)
        causal = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=tgt_in.device), diagonal=1
        )
        tgt_kpm = tgt_in.eq(PAD)
        y = self.drop(self.pos(self.embed(tgt_in) * self.scale))
        self_ws: list[Tensor] = []
        cross_ws: list[Tensor] = []
        for layer in self.decoder_layers:
            y, sw, cw = layer(y, memory, causal, tgt_kpm, src_kpm, need_attn)
            if need_attn:
                self_ws.append(sw)
                cross_ws.append(cw)
        logits = self.out_proj(self.dec_norm(y))
        return logits, self_ws, cross_ws

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        memory, src_kpm, _, _ = self.encode(src)
        logits, _, _ = self.decode(tgt_in, memory, src_kpm)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self, src: Tensor, max_new_tokens: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
        """Autoregressive greedy generation.

        Returns:
            pred: [B, T<=max_new_tokens] generated tokens (EOS included,
                PAD after a sequence finishes).
            confidence: [B] mean max-softmax probability over generated steps
                up to and including EOS.
            memory, src_key_padding_mask, per-layer encoder hiddens — reusable
            for feature extraction without a second encoder pass.
        """
        memory, src_kpm, hiddens, _ = self.encode(src)
        b = src.size(0)
        device = src.device
        tokens = torch.full((b, 1), BOS, dtype=torch.int64, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        conf_sum = torch.zeros(b, device=device)
        conf_cnt = torch.zeros(b, device=device)
        for _ in range(max_new_tokens):
            logits, _, _ = self.decode(tokens, memory, src_kpm)
            probs = logits[:, -1].softmax(dim=-1)
            p, nxt = probs.max(dim=-1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            conf_sum += torch.where(finished, torch.zeros_like(p), p)
            conf_cnt += (~finished).float()
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            finished |= nxt.eq(EOS)
            if bool(finished.all()):
                break
        pred = tokens[:, 1:]
        confidence = (conf_sum / conf_cnt.clamp(min=1.0)).to(torch.float32)
        return pred, confidence, memory, src_kpm, hiddens
