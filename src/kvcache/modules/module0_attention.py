"""Module 0: Attention Mechanism Review — Scaled Dot-Product & Multi-Head Attention.

This module implements the core attention mechanism from scratch using NumPy,
without any deep learning framework. Understanding this is prerequisite to KV Cache.

Key equations:
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

  Multi-Head: split Q,K,V into h heads, attend independently, then concat.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scaled Dot-Product Attention.

    Args:
        Q: (batch, seq_len_q, d_k)
        K: (batch, seq_len_k, d_k)
        V: (batch, seq_len_v, d_v)  — seq_len_k == seq_len_v
        mask: (batch, seq_len_q, seq_len_k) or None; True = masked out.

    Returns:
        output: (batch, seq_len_q, d_v)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores, axis=-1)
    return np.matmul(weights, V)


@dataclass
class MHAConfig:
    d_model: int = 512
    n_heads: int = 8

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads


class MultiHeadAttention:
    """Multi-Head Attention implemented from scratch (NumPy)."""

    def __init__(self, config: MHAConfig, seed: int = 42):
        self.cfg = config
        rng = np.random.default_rng(seed)

        limit = math.sqrt(1.0 / config.d_model)
        self.W_q = rng.uniform(-limit, limit, (config.d_model, config.d_model))
        self.W_k = rng.uniform(-limit, limit, (config.d_model, config.d_model))
        self.W_v = rng.uniform(-limit, limit, (config.d_model, config.d_model))
        self.W_o = rng.uniform(-limit, limit, (config.d_model, config.d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        b, s, _ = x.shape
        return x.reshape(b, s, self.cfg.n_heads, self.cfg.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, n_heads, seq, d_k) -> (batch, seq, d_model)"""
        b, _, s, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, s, self.cfg.d_model)

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None

        Returns:
            output: (batch, seq_len, d_model)
        """
        Q = self._split_heads(x @ self.W_q)
        K = self._split_heads(x @ self.W_k)
        V = self._split_heads(x @ self.W_v)

        b, h, _, _ = Q.shape
        if mask is not None:
            mask = mask[:, None, :, :]
            mask = np.broadcast_to(mask, (b, h, mask.shape[2], mask.shape[3]))

        attn_out = scaled_dot_product_attention(
            Q.reshape(b * h, -1, self.cfg.d_k),
            K.reshape(b * h, -1, self.cfg.d_k),
            V.reshape(b * h, -1, self.cfg.d_k),
            mask.reshape(b * h, mask.shape[2], mask.shape[3]) if mask is not None else None,
        )

        attn_out = attn_out.reshape(b, h, -1, self.cfg.d_k)
        return self._merge_heads(attn_out) @ self.W_o


def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular causal mask: shape (1, seq_len, seq_len)."""
    return np.triu(np.ones((1, seq_len, seq_len), dtype=bool), k=1)


def demo() -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("[bold cyan]Module 0: Attention Mechanism Review[/bold cyan]")
    console.print("─" * 60)

    cfg = MHAConfig(d_model=64, n_heads=4)
    mha = MultiHeadAttention(cfg)

    batch, seq_len = 1, 8
    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch, seq_len, cfg.d_model))

    mask = causal_mask(seq_len)

    console.print(f"[yellow]Input shape:[/yellow]  {x.shape}")
    console.print(f"[yellow]Config:[/yellow] d_model={cfg.d_model}, n_heads={cfg.n_heads}, d_k={cfg.d_k}")
    console.print()

    t0 = time.perf_counter()
    out = mha.forward(x, mask)
    t1 = time.perf_counter()

    console.print(f"[green]Output shape:[/green] {out.shape}")
    console.print(f"[green]Time:[/green]        {(t1-t0)*1000:.3f} ms")
    console.print()

    table = Table(title="Shape Flow (1 layer, 1 batch, seq_len=8)")
    table.add_column("Stage", style="cyan")
    table.add_column("Shape", style="green")
    table.add_row("Input x", str(x.shape))
    table.add_row("Q = x @ W_q", f"{x.shape} @ {mha.W_q.shape} -> {(x @ mha.W_q).shape}")
    table.add_row("Q after split_heads", str(mha._split_heads(x @ mha.W_q).shape))
    table.add_row("QK^T / sqrt(d_k)", f"({batch*cfg.n_heads}, {seq_len}, {cfg.d_k}) @ ... -> ({batch*cfg.n_heads}, {seq_len}, {seq_len})")
    table.add_row("softmax(QK^T) @ V", f"({batch*cfg.n_heads}, {seq_len}, {seq_len}) @ ... -> ({batch*cfg.n_heads}, {seq_len}, {cfg.d_k})")
    table.add_row("After merge + W_o", str(out.shape))
    console.print(table)


if __name__ == "__main__":
    demo()
