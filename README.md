# MiniGPT 🤖

Un modelo de lenguaje construido desde cero en Python + PyTorch.

Aprende a generar texto en español usando dos arquitecturas:
- **v1**: LSTM (más simple, para empezar)
- **v2**: Transformer con Self-Attention (arquitectura GPT real)

---

## 📁 Estructura del proyecto

```
MiniGPT/
├── dataset.txt        ← Datos de entrenamiento (conversaciones)
├── model.py           ← Definición del modelo (LSTM y Transformer)
├── train.py           ← Script de entrenamiento
├── generate.py        ← Generación de texto y chatbot
├── checkpoints/       ← Modelos guardados automáticamente
└── README.md
```

---

## ⚙️ Instalación

```bash
pip install torch numpy tqdm
```

---

## 🚀 Uso rápido

### 1. Entrenar con Transformer (recomendado)
```bash
python train.py --epochs 100
```

### 2. Entrenar con LSTM (más rápido, menos potente)
```bash
python train.py --model lstm --epochs 50
```

### 3. Generar texto
```bash
python generate.py --prompt "Hola" --tokens 200
```

### 4. Chatbot interactivo
```bash
python generate.py --chat
```

---

## 🎛️ Parámetros de entrenamiento

| Parámetro       | Por defecto | Descripción                        |
|-----------------|-------------|------------------------------------|
| `--model`       | transformer | `transformer` o `lstm`             |
| `--epochs`      | 100         | Número de épocas                   |
| `--batch_size`  | 32          | Tamaño del batch                   |
| `--context_len` | 128         | Longitud de contexto en tokens     |
| `--lr`          | 3e-4        | Tasa de aprendizaje                |
| `--embed_dim`   | 128         | Dimensión de embeddings            |
| `--n_heads`     | 4           | Cabezas de atención (solo v2)      |
| `--n_layers`    | 4           | Capas del modelo (solo v2)         |
| `--save_every`  | 10          | Guardar checkpoint cada N épocas   |

---

## 🎛️ Parámetros de generación

| Parámetro       | Por defecto | Descripción                                   |
|-----------------|-------------|-----------------------------------------------|
| `--prompt`      | "Hola"      | Texto inicial                                 |
| `--tokens`      | 200         | Cuántos tokens generar                        |
| `--temperature` | 0.8         | 0.1 = conservador, 1.5 = creativo/aleatorio   |
| `--top_k`       | 40          | Solo considera los K tokens más probables     |
| `--chat`        | -           | Activa el modo chatbot interactivo            |

---

## 🧠 Arquitectura

### MiniGPT v1 (LSTM)
```
Input tokens
    ↓
Embedding (vocab → 128)
    ↓
LSTM (128 → 256, 2 capas)
    ↓
Dropout
    ↓
Linear (256 → vocab)
    ↓
Logits
```

### MiniGPT v2 (Transformer)
```
Input tokens
    ↓
Token Embedding + Positional Embedding
    ↓
[Transformer Block] × N
  ├─ LayerNorm
  ├─ Causal Self-Attention (Multi-Head)
  ├─ LayerNorm
  └─ Feed-Forward (GELU)
    ↓
LayerNorm final
    ↓
Linear (embed → vocab)
    ↓
Logits
```

---

## 💡 Consejos

- **Si el loss no baja**: Reduce el `learning rate` o aumenta el `batch_size`.
- **Si el texto generado es basura**: Entrena más épocas o agrega más datos al `dataset.txt`.
- **Para mejor calidad**: Usa el Transformer (`--model transformer`) con más épocas.
- **CPU lenta**: Reduce `embed_dim` a 64 y `n_layers` a 2.

---

## 📈 Roadmap

- [x] Tokenizador por caracteres
- [x] Modelo LSTM (v1)
- [x] Modelo Transformer con Self-Attention (v2)
- [x] Top-K sampling con temperatura
- [x] Modo chatbot interactivo
- [ ] Tokenizador BPE (como GPT-2)
- [ ] Entrenamiento distribuido (multi-GPU)
- [ ] Fine-tuning con instrucciones
- [ ] Interfaz web con Flask/Gradio
