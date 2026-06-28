"""Module 2: KV Cache Memory Calculation.

Interview-essential skill: given model config and sequence length, compute KV cache size.

Formula (per token, per layer, per attention head group):
  KV_cache = 2 * n_layers * n_kv_heads * d_head * seq_len * batch_size * bytes_per_elem

Key points:
  - The "2" is for K and V (each contributes one tensor).
  - n_kv_heads == n_heads in MHA; smaller in MQA/GQA.
  - bytes_per_elem depends on dtype: 2 (FP16/BF16), 1 (INT8), 0.5 (INT4).
  - This is PER token during decode; the total grows linearly with seq_len.

  Per-token KV cache (bytes) = 2 * n_layers * d_model * bytes_per_elem  (when n_kv_heads*d_head = d_model)
  Total KV cache (bytes)     = per_token_kv * seq_len * batch_size

For MQA/GQA: replace d_model with (n_kv_heads * d_head) which is smaller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DType(Enum):
    FP32 = 4
    FP16 = 2
    BF16 = 2
    INT8 = 1
    INT4 = 0.5


class AttentionType(Enum):
    MHA = "mha"
    MQA = "mqa"
    GQA = "gqa"


@dataclass
class ModelConfig:
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    attn_type: AttentionType = AttentionType.MHA
    dtype: DType = DType.FP16

    def __post_init__(self):
        if self.n_kv_heads is None:
            if self.attn_type == AttentionType.MHA:
                self.n_kv_heads = self.n_heads
            elif self.attn_type == AttentionType.MQA:
                self.n_kv_heads = 1
            else:
                raise ValueError("GQA requires explicit n_kv_heads")
        self.d_head = self.d_model // self.n_heads


def kv_cache_per_token(config: ModelConfig) -> int:
    """Bytes per token (single batch element) for KV cache across all layers."""
    kv_elems_per_token = 2 * config.n_layers * config.n_kv_heads * config.d_head
    return int(kv_elems_per_token * config.dtype.value)


def kv_cache_total(config: ModelConfig, seq_len: int, batch_size: int = 1) -> int:
    """Total KV cache bytes for given sequence length and batch size."""
    return kv_cache_per_token(config) * seq_len * batch_size


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024
    return f"{v:.2f} PB"


def demo() -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    console.print(Panel(
        "[bold cyan]Module 2: KV Cache Memory Calculator[/bold cyan]\n\n"
        "Formula: KV = 2 × n_layers × n_kv_heads × d_head × seq_len × batch × bytes\n"
        "  • 2 = K + V\n"
        "  • n_kv_heads = n_heads (MHA) / 1 (MQA) / grouped (GQA)",
        border_style="cyan",
    ))

    models = [
        ModelConfig("GPT-2 (124M)",       n_layers=12,  d_model=768,  n_heads=12,  dtype=DType.FP16),
        ModelConfig("LLaMA-2 7B",         n_layers=32,  d_model=4096, n_heads=32,  dtype=DType.FP16),
        ModelConfig("LLaMA-2 13B",        n_layers=40,  d_model=5120, n_heads=40,  dtype=DType.FP16),
        ModelConfig("LLaMA-2 70B (GQA)",  n_layers=80,  d_model=8192, n_heads=64,  n_kv_heads=8,  attn_type=AttentionType.GQA, dtype=DType.FP16),
        ModelConfig("Mistral-7B (GQA)",   n_layers=32,  d_model=4096, n_heads=32,  n_kv_heads=8,  attn_type=AttentionType.GQA, dtype=DType.FP16),
        ModelConfig("GPT-3 175B (MQA)",   n_layers=96,  d_model=12288,n_heads=96,  n_kv_heads=1,  attn_type=AttentionType.MQA, dtype=DType.FP16),
    ]

    seq_lens = [128, 512, 2048, 4096, 8192, 32768]

    table = Table(title="KV Cache Size per Token (FP16/BF16, batch=1)")
    table.add_column("Model", style="cyan")
    table.add_column("Layers", justify="right")
    table.add_column("d_model", justify="right")
    table.add_column("n_heads", justify="right")
    table.add_column("n_kv_heads", justify="right")
    table.add_column("Bytes/token", justify="right", style="green")
    table.add_column("Tokens/GB", justify="right", style="yellow")

    for m in models:
        per_tok = kv_cache_per_token(m)
        tok_per_gb = int((1024**3) / per_tok)
        table.add_row(
            m.name, str(m.n_layers), str(m.d_model), str(m.n_heads),
            str(m.n_kv_heads), format_bytes(per_tok), f"{tok_per_gb:,}",
        )
    console.print(table)

    console.print()
    table2 = Table(title="Total KV Cache Size by Context Length (batch=1, FP16)")
    table2.add_column("Model", style="cyan")
    for sl in seq_lens:
        table2.add_column(f"{sl} ctx", justify="right", style="green")

    for m in models:
        row = [m.name]
        for sl in seq_lens:
            row.append(format_bytes(kv_cache_total(m, sl)))
        table2.add_row(*row)
    console.print(table2)

    console.print()
    console.print("[bold]Key takeaways:[/bold]")
    console.print(
        "  1. KV cache grows [green]linearly[/green] with context length (not quadratic).\n"
        "  2. MQA/GQA reduces KV cache by factor n_heads/n_kv_heads (e.g., 8x for LLaMA-2-70B).\n"
        "  "
        "  3. Quantization (INT8/INT4) linearly reduces KV cache size.\n"
        "  4. For LLaMA-2-7B at 4k context: ~256 MB KV cache; at 32k: ~2 GB.\n"
        "  5. For GPT-3 175B at 2k context (MQA): ~482 MB per request; batching multiplies this."
    )

    console.print()
    console.print("[bold]Memory estimator (interactive):[/bold]")
    m = models[3]
    console.print(f"  Example: [cyan]{m.name}[/cyan] at 4096 context, batch=32:")
    total = kv_cache_total(m, 4096, batch_size=32)
    console.print(f"    KV cache = {format_bytes(total)}")
    console.print(f"    Per-token = {format_bytes(kv_cache_per_token(m))}")
    m_int8 = ModelConfig(m.name + " INT8", m.n_layers, m.d_model, m.n_heads, m.n_kv_heads, m.attn_type, DType.INT8)
    total_int8 = kv_cache_total(m_int8, 4096, batch_size=32)
    console.print(f"    INT8 KV  = {format_bytes(total_int8)}  ({total/total_int8:.1f}x reduction)")


if __name__ == "__main__":
    demo()
