"""
generate.py — Generación de texto con MiniGPT

Uso:
    # Generación libre
    python generate.py --prompt "Hola" --tokens 200

    # Modo chatbot interactivo
    python generate.py --chat

    # Especificar checkpoint
    python generate.py --checkpoint checkpoints/best_model.pt --chat
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F

from model import MiniGPTv1, MiniGPTv2


# ──────────────────────────────────────────────
#  Cargar tokenizer
# ──────────────────────────────────────────────

def load_tokenizer(checkpoint_dir: str):
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"No se encontró el tokenizer en {tok_path}. ¿Entrenaste el modelo?")
    with open(tok_path, encoding="utf-8") as f:
        data = json.load(f)

    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}

    class Tokenizer:
        def __init__(self):
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)

        def encode(self, s):
            return [self.stoi[c] for c in s if c in self.stoi]

        def decode(self, tokens):
            return ''.join(self.itos.get(i, '?') for i in tokens)

    return Tokenizer()


# ──────────────────────────────────────────────
#  Cargar modelo desde checkpoint
# ──────────────────────────────────────────────

def load_model(checkpoint_path: str, tokenizer, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model_type = args.get("model", "transformer")

    if model_type == "transformer":
        model = MiniGPTv2(
            vocab_size=tokenizer.vocab_size,
            context_len=args.get("context_len", 128),
            embed_dim=args.get("embed_dim", 128),
            n_heads=args.get("n_heads", 4),
            n_layers=args.get("n_layers", 4),
        )
    else:
        model = MiniGPTv1(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.get("embed_dim", 128),
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss", "?")
    print(f"[Model] Cargado '{model_type}' — época {epoch}, loss {loss:.4f}")
    return model, model_type, args


# ──────────────────────────────────────────────
#  Generación con temperatura y top-k sampling
# ──────────────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    model_type: str,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    context_len: int = 128,
    device: str = "cpu",
):
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]  # fallback si el prompt tiene caracteres desconocidos

    x = torch.tensor([tokens], dtype=torch.long, device=device)
    hidden = None

    generated = list(tokens)

    for _ in range(max_new_tokens):
        # Limitar contexto
        if model_type == "transformer":
            x_feed = x[:, -context_len:]
            logits = model(x_feed)          # (1, T, vocab)
            logits = logits[:, -1, :]       # último token
        else:
            logits, hidden = model(x, hidden)
            logits = logits[:, -1, :]       # último token

        # Temperatura
        logits = logits / temperature

        # Top-k sampling
        if top_k > 0:
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_vals[:, -1].unsqueeze(-1)
            logits[logits < threshold] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        generated.append(next_token.item())
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(generated)


# ──────────────────────────────────────────────
#  Modo chatbot
# ──────────────────────────────────────────────

def chat_mode(model, model_type, tokenizer, args_model, device):
    context_len = args_model.get("context_len", 128)
    print("\n" + "="*55)
    print("  🤖  MiniGPT Chatbot  —  escribe 'salir' para terminar")
    print("="*55 + "\n")

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chatbot] ¡Hasta luego!")
            break

        if user_input.lower() in ("salir", "exit", "quit"):
            print("[Chatbot] ¡Hasta luego!")
            break
        if not user_input:
            continue

        # Formatear como dataset de chat
        prompt = f"Usuario: {user_input}\nBot:"

        response_raw = generate(
            model=model,
            model_type=model_type,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.75,
            top_k=30,
            context_len=context_len,
            device=device,
        )

        # Extraer solo la respuesta del bot
        if "Bot:" in response_raw:
            bot_part = response_raw.split("Bot:")[-1]
            # Cortar en el siguiente turno de usuario
            if "Usuario:" in bot_part:
                bot_part = bot_part.split("Usuario:")[0]
            response = bot_part.strip()
        else:
            response = response_raw[len(prompt):].strip()

        print(f"Bot: {response}\n")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genera texto con MiniGPT")
    parser.add_argument("--checkpoint",     type=str,   default="checkpoints/best_model.pt")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--prompt",         type=str,   default="Hola")
    parser.add_argument("--tokens",         type=int,   default=200,  help="Tokens a generar")
    parser.add_argument("--temperature",    type=float, default=0.8,  help="Temperatura (0.1=conservador, 1.5=creativo)")
    parser.add_argument("--top_k",          type=int,   default=40,   help="Top-K sampling")
    parser.add_argument("--chat",           action="store_true",      help="Modo chatbot interactivo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Verificar checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"[Error] No se encontró el checkpoint: {args.checkpoint}")
        print("  → Entrena primero con:  python train.py")
        return

    tokenizer = load_tokenizer(args.checkpoint_dir)
    model, model_type, args_model = load_model(args.checkpoint, tokenizer, device)

    if args.chat:
        chat_mode(model, model_type, tokenizer, args_model, device)
    else:
        print(f"\n[Generando] Prompt: '{args.prompt}'")
        print("-" * 40)
        output = generate(
            model=model,
            model_type=model_type,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            context_len=args_model.get("context_len", 128),
            device=str(device),
        )
        print(output)


if __name__ == "__main__":
    main()
