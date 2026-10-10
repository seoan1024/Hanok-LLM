"""Hanok's architecture, moved from the v4 implementation without shape changes."""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _linear_forward(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply a linear layer and its optional low-rank adapter."""
    output = layer(x)
    lora_a = getattr(layer, "lora_A", None)
    if lora_a is not None:
        output = output + F.linear(F.linear(x, lora_a), layer.lora_B) * layer.lora_scale
    return output

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Accumulate the RMS in fp32 even when the model weights/activations
        # use bf16. Low precision reductions in a 1,920-wide hidden state can
        # otherwise inject avoidable noise at every block.
        input_dtype = x.dtype
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(
            x_fp32.square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(input_dtype)

def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _, head_dim_2 = cos.shape
    head_dim = head_dim_2 * 2

    input_dtype = x.dtype
    x = x.float()
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    cos = cos.float().unsqueeze(0).unsqueeze(0)
    sin = sin.float().unsqueeze(0).unsqueeze(0)

    return torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
    ).to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        hidden = F.silu(_linear_forward(self.w1, x)) * _linear_forward(self.w3, x)
        return _linear_forward(self.w2, hidden)

class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        f_cos: torch.Tensor,
        f_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, s, d = x.shape

        q = _linear_forward(self.wq, x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = _linear_forward(self.wk, x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = _linear_forward(self.wv, x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        new_kv = (k.detach(), v.detach())

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and s > 1)
        )

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return _linear_forward(self.wo, out), new_kv

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        f_cos: torch.Tensor,
        f_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv

class KoreanLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128256,
        pad_token_id: int = 128004,
        dim: int = 1920,
        n_layers: int = 20,
        n_heads: int = 10,
        max_seq_len: int = 4096
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, int(dim * 2.5))
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight

        f_cos, f_sin = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)

        self._init_weights()

    def enable_lora(self, rank: int = 8, alpha: Optional[float] = None) -> None:
        """Freeze base weights and attach trainable low-rank adapters to linears."""
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        alpha = float(alpha if alpha is not None else rank * 2)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            # The output projection shares its base weight with the embedding.
            # Keeping it tied avoids an extra vocabulary-sized adapter and
            # preserves compatibility with the standard tied Llama export.
            if module is self.output:
                continue
            if hasattr(module, "lora_A"):
                continue
            a = nn.Parameter(torch.empty(rank, module.in_features, device=module.weight.device,
                                         dtype=module.weight.dtype))
            b = nn.Parameter(torch.zeros(module.out_features, rank, device=module.weight.device,
                                         dtype=module.weight.dtype))
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)
            module.register_parameter("lora_A", a)
            module.register_parameter("lora_B", b)
            module.lora_scale = alpha / rank
            a.requires_grad_(True)
            b.requires_grad_(True)

        self.lora_rank = rank
        self.lora_alpha = alpha

    @staticmethod
    def merged_linear_weight(module: nn.Linear) -> torch.Tensor:
        weight = module.weight
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None:
            return weight
        return weight + (module.lora_B @ lora_a) * module.lora_scale

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _get_freqs(self, f: torch.Tensor, start: int, length: int) -> torch.Tensor:
        return f[start:start + length]

    def reset_rope_cache(self) -> None:
        """Rebuild deterministic RoPE tables in fp32 on the current device.

        Older checkpoints may contain bf16-rounded frequency buffers because
        the training path converted the entire module to bf16. Those buffers
        are derived values, so restore them rather than treating them as
        learned checkpoint state.
        """
        end = self.f_cos.shape[0]
        cos, sin = precompute_freqs_cis(self.head_dim, end)
        self.f_cos = cos.to(device=self.embed.weight.device, dtype=torch.float32)
        self.f_sin = sin.to(device=self.embed.weight.device, dtype=torch.float32)

    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        b, s = tokens.shape
        if tokens.dtype != torch.long:
            raise TypeError("tokens must contain integer token IDs (torch.long)")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError(f"token IDs must be in [0, {self.vocab_size})")
        if labels is not None and labels.shape != tokens.shape:
            raise ValueError("labels must have the same shape as tokens")
        x = self.embed(tokens)

        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0][0] is not None:
            start_pos = kv_caches[0][0].shape[2]

        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)
        if f_cos.shape[0] != s:
            raise ValueError(
                f"Sequence with cache is longer than the RoPE context ({self.f_cos.shape[0]} tokens)"
            )

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if self.training:
                x, kv = checkpoint(
                    layer, x, f_cos, f_sin, None,
                    use_reentrant=False
                )
            else:
                kv_cache = kv_caches[i] if kv_caches else None
                x, kv = layer(x, f_cos, f_sin, kv_cache=kv_cache)

            new_kv_caches.append(kv)

        x = self.norm(x)
        logits = _linear_forward(self.output, x)

        loss = None
        if labels is not None:
            # ``reduction='mean'`` returns NaN when a malformed batch has no
            # target tokens (every label is -100).  Preserve mean-loss
            # semantics for valid batches while making that case a zero-loss
            # batch with zero gradients.
            token_loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
                reduction='sum'
            )
            target_count = (labels[..., 1:] != -100).sum().clamp_min(1)
            loss = token_loss / target_count

        return logits, loss, new_kv_caches

