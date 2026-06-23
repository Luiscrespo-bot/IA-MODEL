"""
train.py — Entrenamiento de MiniGPT

Uso:
    python train.py                      # Entrena con Transformer (v2, por defecto)
    python train.py --model lstm         # Entrena con LSTM (v1)
    python train.py --epochs 50 --lr 3e-4
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import MiniGPTv1, MiniGPTv2


# ──────────────────────────────────────────────
#  Tokenizador por caracteres
# ──────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        print(f"[Tokenizer] Vocabulario: {self.vocab_size} caracteres únicos")

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, tokens):
        return ''.join(self.itos.get(i, '?') for i in tokens)


# ──────────────────────────────────────────────
#  Dataset de secuencias
# ──────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, tokens: list, context_len: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return max(0, len(self.tokens) - self.context_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.context_len + 1]
        x = chunk[:-1]   # entrada
        y = chunk[1:]     # objetivo (desplazado 1)
        return x, y


# ──────────────────────────────────────────────
#  Funciones de entrenamiento
# ──────────────────────────────────────────────

def train_epoch_transformer(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)                        # (B, T, vocab)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch_lstm(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)                     # (B, T, vocab)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrena MiniGPT")
    parser.add_argument("--model",       type=str,   default="transformer", choices=["transformer", "lstm"],
                        help="Arquitectura: 'transformer' (GPT) o 'lstm'")
    parser.add_argument("--dataset",     type=str,   default="dataset.txt")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--context_len", type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--embed_dim",   type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every",  type=int,   default=10,
                        help="Guardar checkpoint cada N épocas")
    args = parser.parse_args()

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Usando: {device}")

    # Dataset
    with open(args.dataset, encoding="utf-8") as f:
        text = f.read()
    print(f"[Dataset] {len(text):,} caracteres cargados")

    # Tokenizer
    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)

    # Guardar tokenizer para inferencia
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    import json
    tok_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump({"stoi": tokenizer.stoi, "itos": {str(k): v for k, v in tokenizer.itos.items()}}, f, ensure_ascii=False)
    print(f"[Tokenizer] Guardado en {tok_path}")

    # Dataset y DataLoader
    dataset = TextDataset(tokens, args.context_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[Dataset] {len(dataset):,} muestras, {len(loader)} batches por época")

    # Modelo
    if args.model == "transformer":
        model = MiniGPTv2(
            vocab_size=tokenizer.vocab_size,
            context_len=args.context_len,
            embed_dim=args.embed_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(device)
        train_epoch = train_epoch_transformer
    else:
        model = MiniGPTv1(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=256,
        ).to(device)
        train_epoch = train_epoch_lstm

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {args.model.upper()} — {total_params:,} parámetros")

    # Optimizador y función de pérdida
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    # ── Bucle de entrenamiento ──
    best_loss = float('inf')
    print("\n" + "="*50)
    print(f"  Iniciando entrenamiento: {args.epochs} épocas")
    print("="*50 + "\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Época {epoch:3d}/{args.epochs}  |  Loss: {loss:.4f}  |  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}  |  {elapsed:.1f}s")

        # Guardar mejor modelo
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss": loss,
                "args": vars(args),
            }, best_path)

        # Checkpoint periódico
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> Checkpoint guardado: {ckpt_path}")

    print(f"\nEntrenamiento completado! Mejor loss: {best_loss:.4f}")
    print(f"  Modelo guardado en: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
