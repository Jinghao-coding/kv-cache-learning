"""Module 1: KV Cache — Basic Principle and Implementation.

Core idea of KV Cache:
  During autoregressive generation, at step t we already computed K, V for tokens 0..t-1.
  These do NOT change in future steps. Instead of recomputing them every time, we CACHE them.

Naive generation (step t):
  Feed [token_0, ..., token_t] through the entire model every step.
  -> Each layer recomputes K,V for ALL t+1 tokens: O(n^2) total FLOPs.

KV Cache generation:
  Prefill:  feed all prompt tokens once, cache K,V per layer.
  Decode:   feed only the NEW token each step; for each layer compute q_new, k_new, v_new,
            append k_new, v_new to cache, attend q_new to ALL cached K,V.
  -> Per-layer K/V projections saved for past tokens.
  -> Q@K^T cost is the same order, but large linear projections are skipped.

Key insight:
  KV Cache exploits the fact that in causal attention, past K,V are IMMUTABLE once computed.
  The saving grows with sequence length and model width.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kvcache.modules.module0_attention import MHAConfig, softmax


@dataclass
class KVCache:
    """Per-layer KV cache: K, V have shape (batch*n_heads, seq_len, d_k)."""

    K: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> None:
        if self.K is None:
            self.K = k_new
            self.V = v_new
        else:
            self.K = np.concatenate([self.K, k_new], axis=1)
            self.V = np.concatenate([self.V, v_new], axis=1)

    @property
    def seq_len(self) -> int:
        return 0 if self.K is None else self.K.shape[1]


class CachedMHA:
    """Multi-Head Attention with prefill and incremental-decode paths."""

    def __init__(self, config: MHAConfig, seed: int = 42):
        self.cfg = config
        rng = np.random.default_rng(seed)
        limit = math.sqrt(1.0 / config.d_model)
        self.W_q = rng.uniform(-limit, limit, (config.d_model, config.d_model))
        self.W_k = rng.uniform(-limit, limit, (config.d_model, config.d_model))
        self.W_v = rng.uniform(-limit, limit, (config.d_model, config.d_model))
        self.W_o = rng.uniform(-limit, limit, (config.d_model, config.d_model))

    def _project_qkv(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project x to Q, K, V and split heads. x: (b, s, d) -> flat (b*h, s, d_k)."""
        b, s, _ = x.shape
        h, d_k = self.cfg.n_heads, self.cfg.d_k
        Q = (x @ self.W_q).reshape(b, s, h, d_k).transpose(0, 2, 1, 3).reshape(b * h, s, d_k)
        K = (x @ self.W_k).reshape(b, s, h, d_k).transpose(0, 2, 1, 3).reshape(b * h, s, d_k)
        V = (x @ self.W_v).reshape(b, s, h, d_k).transpose(0, 2, 1, 3).reshape(b * h, s, d_k)
        return Q, K, V

    def _merge(self, flat: np.ndarray, b: int) -> np.ndarray:
        """(b*h, s, d_k) -> (b, s, d)."""
        h, d_k = self.cfg.n_heads, self.cfg.d_k
        s = flat.shape[1]
        return flat.reshape(b, h, s, d_k).transpose(0, 2, 1, 3).reshape(b, s, self.cfg.d_model)

    def _attend(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        d_k = self.cfg.d_k
        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        return softmax(scores, axis=-1) @ V

    def forward_full(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        b, s, _ = x.shape
        Q, K, V = self._project_qkv(x)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, :, :]
            h = self.cfg.n_heads
            attn_mask = np.broadcast_to(attn_mask, (b, h, s, s)).reshape(b * h, s, s)
        out = self._attend(Q, K, V, attn_mask)
        return self._merge(out, b) @ self.W_o

    def forward_prefill(self, x: np.ndarray, cache: KVCache, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Prefill: process full sequence, populate cache with K,V, return output."""
        b, s, _ = x.shape
        Q, K, V = self._project_qkv(x)
        attn_mask = None
        if mask is not None:
            h = self.cfg.n_heads
            attn_mask = mask[:, None, :, :]
            attn_mask = np.broadcast_to(attn_mask, (b, h, s, s)).reshape(b * h, s, s)
        out = self._attend(Q, K, V, attn_mask)
        cache.K = K
        cache.V = V
        return self._merge(out, b) @ self.W_o

    def forward_step(self, x_new: np.ndarray, cache: KVCache) -> np.ndarray:
        """Decode one step: x_new is (b,1,d); append k,v to cache; attend to all cached."""
        b, _, _ = x_new.shape
        q_new, k_new, v_new = self._project_qkv(x_new)
        K_all = k_new if cache.K is None else np.concatenate([cache.K, k_new], axis=1)
        V_all = v_new if cache.V is None else np.concatenate([cache.V, v_new], axis=1)
        out = self._attend(q_new, K_all, V_all)
        cache.K = K_all
        cache.V = V_all
        return self._merge(out, b) @ self.W_o


class SimpleTransformer:
    """A minimal transformer (embedding -> N x [MHA + residual] -> lm_head) for KV-cache demo.

    No MLP/LayerNorm to keep the code focused on the KV-cache mechanism.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        vocab_size: int = 1000,
        seed: int = 42,
    ):
        self.cfg = MHAConfig(d_model=d_model, n_heads=n_heads)
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self.embedding = rng.standard_normal((vocab_size, d_model)) * 0.02
        self.layers = [CachedMHA(self.cfg, seed=seed + i + 1) for i in range(n_layers)]
        self.lm_head = rng.standard_normal((d_model, vocab_size)) * 0.02

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        return self.embedding[ids]

    @staticmethod
    def _causal_mask(seq_len: int) -> np.ndarray:
        return np.triu(np.ones((1, seq_len, seq_len), dtype=bool), k=1)

    def generate_naive(self, prompt: np.ndarray, max_new: int) -> tuple[np.ndarray, float]:
        """Naive: full forward pass over entire sequence each decode step."""
        b, _ = prompt.shape
        ids = prompt.copy()
        rng = np.random.default_rng(123)
        t0 = time.perf_counter()
        for _ in range(max_new):
            x = self._embed(ids)
            seq = ids.shape[1]
            mask = self._causal_mask(seq)
            for layer in self.layers:
                x = layer.forward_full(x, mask) + x
            logits = x[:, -1, :] @ self.lm_head
            probs = softmax(logits, axis=-1)
            next_ids = np.array([[rng.choice(self.vocab_size, p=probs[i])] for i in range(b)])
            ids = np.concatenate([ids, next_ids], axis=1)
        return ids, time.perf_counter() - t0

    def generate_kv_cache(self, prompt: np.ndarray, max_new: int) -> tuple[np.ndarray, float]:
        """KV-Cache: prefill once, then decode one token at a time."""
        b, _ = prompt.shape
        ids = prompt.copy()
        rng = np.random.default_rng(123)
        t0 = time.perf_counter()

        caches: list[KVCache] = [KVCache() for _ in range(self.n_layers)]

        # Prefill
        x = self._embed(ids)
        seq = ids.shape[1]
        mask = self._causal_mask(seq)
        for li, layer in enumerate(self.layers):
            x = layer.forward_prefill(x, caches[li], mask) + x

        logits = x[:, -1, :] @ self.lm_head
        probs = softmax(logits, axis=-1)
        next_ids = np.array([[rng.choice(self.vocab_size, p=probs[i])] for i in range(b)])
        ids = np.concatenate([ids, next_ids], axis=1)

        # Decode
        for _ in range(max_new - 1):
            x_new = self._embed(next_ids)
            for li, layer in enumerate(self.layers):
                x_new = layer.forward_step(x_new, caches[li]) + x_new
            logits = x_new[:, 0, :] @ self.lm_head
            probs = softmax(logits, axis=-1)
            next_ids = np.array([[rng.choice(self.vocab_size, p=probs[i])] for i in range(b)])
            ids = np.concatenate([ids, next_ids], axis=1)

        return ids, time.perf_counter() - t0


def demo() -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    console.print(Panel(
        "[bold cyan]Module 1: KV Cache — Basic Principle[/bold cyan]\n\n"
        "Naive: recomputes Q,K,V for ALL tokens each decode step.\n"
        "KV Cache: caches K,V after prefill; decode step only processes 1 new token.",
        border_style="cyan",
    ))

    d_model, n_heads, n_layers, vocab = 64, 4, 4, 1000
    prompt_len = 16
    gen_len = 64

    console.print(f"[yellow]Config:[/yellow] d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    console.print(f"[yellow]Sequence:[/yellow] prompt={prompt_len} tokens, generate={gen_len} tokens\n")

    model = SimpleTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers, vocab_size=vocab, seed=42)
    prompt = np.arange(prompt_len)[None, :]

    with console.status("[bold]Running naive generation..."):
        ids_naive, t_naive = model.generate_naive(prompt, gen_len)
    with console.status("[bold]Running KV-cache generation..."):
        ids_cache, t_cache = model.generate_kv_cache(prompt, gen_len)

    match = np.array_equal(ids_naive, ids_cache)

    table = Table(title="Generation Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Time (ms)", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="yellow")
    table.add_column("Output match?", justify="center")
    table.add_row("Naive (recompute all)", f"{t_naive*1000:.2f}", "1.00x", "—")
    table.add_row("KV Cache", f"{t_cache*1000:.2f}", f"{t_naive/t_cache:.2f}x", "✅" if match else "❌")
    console.print(table)

    if not match:
        console.print("[red]WARNING: outputs differ — bug in KV cache implementation![/red]")
        return

    console.print()
    console.print("[bold green]✅ Naive and KV-cache produce IDENTICAL outputs, as expected.[/bold green]")
    console.print()

    console.print("[bold]Why does KV Cache help?[/bold]")
    console.print(
        "In each decode step for layer l:\n"
        "  • Naive: projects Q,K,V for the FULL sequence length t → O(t·d²) FLOPs in projections.\n"
        "  • Cache: projects q,k,v for the 1 new token → O(d²) FLOPs in projections; then attends\n"
        "    q_new to all cached K,V (O(t·d_k) matmul — the same attention score cost).\n\n"
        "The [green]savings come from linear projections[/green] (W_q, W_k, W_v, W_o) which are d_model×d_model.\n"
        "At d_model=4096 and t=4096, naive does ~3·4096²·4096 ≈ 206 GFLOPs per layer per step in K/V projections\n"
        "alone, while cache does ~3·4096² ≈ 50 MFLOPs — a [bold]~4000x[/bold] saving on those projections."
    )

    console.print()
    table2 = Table(title="Prefill vs Decode")
    table2.add_column("Phase", style="cyan")
    table2.add_column("Q shape", justify="center")
    table2.add_column("K,V shape", justify="center")
    table2.add_column("Projection cost", justify="right", style="yellow")
    table2.add_column("Notes")
    table2.add_row(
        "Prefill", f"({prompt_len}, {d_model})", f"({prompt_len}, {d_model})",
        f"O({prompt_len}·{d_model}²)", "Parallel; populates cache",
    )
    table2.add_row(
        "Decode/step", f"(1, {d_model})", f"(t+1, {d_model}) [cached]",
        f"O({d_model}²)", f"One new token; {gen_len} steps total",
    )
    console.print(table2)


if __name__ == "__main__":
    demo()
