"""
model.py — Definición de los modelos MiniGPT

Contiene dos versiones:
  - MiniGPTv1: Basado en LSTM (más simple, para empezar)
  - MiniGPTv2: Basado en Transformer (arquitectura GPT real)
"""

import torch
import torch.nn as nn
import math


# ──────────────────────────────────────────────
#  Versión 1: LSTM
# ──────────────────────────────────────────────

class MiniGPTv1(nn.Module):
    """
    Modelo de lenguaje simple basado en LSTM.
    Útil para aprender los conceptos antes de usar Transformers.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        emb = self.dropout(self.embedding(x))         # (batch, seq, embed)
        out, hidden = self.lstm(emb, hidden)           # (batch, seq, hidden)
        logits = self.fc(self.dropout(out))            # (batch, seq, vocab)
        return logits, hidden


# ──────────────────────────────────────────────
#  Versión 2: Transformer (estilo GPT)
# ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention con máscara causal (solo ve tokens anteriores)."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Calcular Q, K, V de una vez
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Atención escalada
        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale     # (B, heads, T, T)

        # Máscara causal: no ver hacia adelante
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ v                                # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        return self.resid_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Un bloque Transformer: Atención + Feed-Forward + Layer Norm."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection
        x = x + self.ff(self.ln2(x))      # residual connection
        return x


class MiniGPTv2(nn.Module):
    """
    Modelo de lenguaje basado en Transformer (arquitectura GPT).
    Esta es la versión real — usa self-attention en vez de LSTM.
    """

    def __init__(
        self,
        vocab_size: int,
        context_len: int = 256,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_len = context_len

        # Embeddings de token y posición
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        # Bloques Transformer apilados
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Inicialización de pesos (mejora la convergencia)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.context_len, f"Secuencia demasiado larga: {T} > {self.context_len}"

        pos = torch.arange(T, device=x.device).unsqueeze(0)      # (1, T)
        emb = self.drop(self.token_emb(x) + self.pos_emb(pos))   # (B, T, C)

        out = self.blocks(emb)                                     # (B, T, C)
        out = self.ln_f(out)
        logits = self.head(out)                                    # (B, T, vocab)
        return logits
