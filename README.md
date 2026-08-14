# StellarAI

**A lightweight multimodal language model trained from scratch — ~50M parameters, runs on CPU.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)

---

## What is StellarAI?

StellarAI is a from-scratch, bilingual (Chinese + English) causal language model designed for **learning and prototyping**. With just ~50M parameters (0.05B), it runs entirely on CPU with 4GB RAM — no GPU required.

It features:
- A 4-layer Transformer text encoder with RoPE positional encoding
- A CNN + ViT hybrid vision encoder for multimodal input
- A cross-attention fusion block for text-image integration
- A built-in **plugin system** (calculator, translator, knowledge base, text tools, time queries)
- BPE tokenizer with mixed Chinese-English support

> **Looking for the pretrained weights?** Download from [Releases](https://github.com/amtstudio/StellarAI/releases) or [Hugging Face Hub](https://huggingface.co/amtstudio/stellarai-tiny).

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/amtstudio/StellarAI.git
cd StellarAI
pip install -r requirements.txt
```

### 2. Download Pretrained Weights

Download `stellar_final.pt` and `tokenizer.json` from the [Releases page](https://github.com/amtstudio/StellarAI/releases) and place them in a directory (e.g., `checkpoints/`).

### 3. Run Inference

```python
from stellarai import StellarTorch, SimpleTokenizer, StellarConfig

# Load model
cfg = StellarConfig.tiny()
model = StellarTorch(cfg)
model.load_state_dict(torch.load("checkpoints/stellar_final.pt", map_location="cpu"))
model.eval()

# Load tokenizer
tokenizer = SimpleTokenizer(cfg.vocab_size)
tokenizer.load("checkpoints/tokenizer.json")

# Generate
prompt = "Artificial intelligence is"
ids = tokenizer.encode(prompt, add_bos=True)
input_ids = torch.tensor([ids], dtype=torch.long)

for _ in range(50):
    logits = model(input_ids)["logits"][0, -1, :cfg.vocab_size]
    next_id = torch.multinomial(torch.softmax(logits / 0.7, dim=-1), 1)
    input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
    if next_id.item() == tokenizer.eos_token_id:
        break

print(tokenizer.decode(input_ids[0].tolist()))
```

### 4. Or Use the Hugging Face API

If you prefer the HF ecosystem, grab the model from [Hugging Face Hub](https://huggingface.co/amtstudio/stellarai-tiny):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "amtstudio/stellarai-tiny", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "amtstudio/stellarai-tiny", trust_remote_code=True
)
```

---

## Project Structure

```
StellarAI/
├── stellarai/                  # Core library
│   ├── config.py               # Model configuration (StellarConfig)
│   ├── model.py                # NumPy reference model
│   ├── model_torch.py          # PyTorch model (StellarTorch)
│   ├── tokenizer.py            # BPE tokenizer (SimpleTokenizer)
│   ├── inference.py            # Inference helpers
│   ├── training.py             # Training utilities
│   ├── vision.py               # Vision encoder
│   └── plugins/                # Plugin system
│       ├── base.py             # Plugin base class & manager
│       ├── calculator.py       # Math expression evaluator
│       ├── knowledge.py        # Built-in knowledge base
│       ├── translator.py       # Basic translation
│       ├── text_tools.py       # Text manipulation utilities
│       └── time_tool.py        # Date/time queries
├── data/                       # Training data
│   ├── train_corpus.txt        # Hand-crafted training corpus
│   ├── build_corpus.py         # Corpus builder
│   ├── generate_corpus.py      # Synthetic data generator
│   ├── download_hf_datasets.py # HF dataset downloader
│   └── merge_corpus.py         # Corpus merger
├── examples/                   # Usage examples
│   └── quick_start.py          # Quick start demo
├── train.py                    # Main training script
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

---

## Training Your Own Model

### 1. Prepare Training Data

Place your text corpus in `data/train_corpus.txt` (one sentence per line). The corpus should be bilingual (Chinese + English) for best results.

### 2. Start Training

```bash
python train.py \
    --data data/train_corpus.txt \
    --output outputs/stellar_pt \
    --steps 5000 \
    --batch_size 4 \
    --seq_len 128 \
    --lr 3e-4
```

Key arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to training corpus | `data/train_corpus.txt` |
| `--output` | Output directory for checkpoints | `outputs/stellar_pt` |
| `--steps` | Total training steps | `5000` |
| `--batch_size` | Batch size | `4` |
| `--seq_len` | Sequence length | `128` |
| `--lr` | Peak learning rate | `3e-4` |
| `--tokenizer` | Path to existing tokenizer (skip BPE training) | `None` |
| `--resume` | Path to checkpoint for resuming training | `None` |

### 3. Monitor Training

Training logs are saved to `outputs/train_log.txt` and sample generations are printed every 200 steps.

---

## Plugin System

StellarAI includes a plugin system inspired by function calling. The model can invoke tools using the `[TOOL:name]` syntax:

```
[TOOL:calc] 2**10 + 5*3 [/TOOL]
[TOOL:translate] en:zh Hello world [/TOOL]
[TOOL:time] now [/TOOL]
[TOOL:knowledge] What is a Transformer? [/TOOL]
```

### Available Plugins

| Plugin | Description | Example |
|--------|-------------|---------|
| `calc` | Math expression evaluation | `[TOOL:calc] 2**10 [/TOOL]` |
| `translate` | Basic EN↔ZH translation | `[TOOL:translate] en:zh Hello [/TOOL]` |
| `knowledge` | Built-in knowledge base | `[TOOL:knowledge] Transformer [/TOOL]` |
| `text_tools` | Text manipulation | `[TOOL:text_tools] reverse hello [/TOOL]` |
| `time` | Date/time queries | `[TOOL:time] now [/TOOL]` |

### Using Plugins in Code

```python
from stellarai.plugins import create_default_manager

manager = create_default_manager()
result = manager.execute("[TOOL:calc] 2**10 + 5*3 [/TOOL]")
print(result)  # [TOOL_RESULT]1039[/TOOL_RESULT]
```

---

## Architecture Overview

```
StellarAI-Tiny (~50M Parameters)
├── Embedding              vocab(32000) x d_model(384)
├── Text Transformer (4 layers)
│   ├── Multi-Head Self-Attention (6 heads, RoPE)
│   └── FFN (GELU, intermediate=1536)
├── Vision Encoder (CNN + ViT)
│   ├── CNN Feature Extractor (4 layers)
│   └── ViT Transformer (2 layers, 6 heads)
├── Fusion Block (1 layer)
│   ├── Self-Attention + Cross-Attention
│   └── FFN (1536)
└── LM Head (tied weights)
```

| Config | Value |
|--------|-------|
| `d_model` | 384 |
| `num_hidden_layers` | 4 |
| `num_attention_heads` | 6 |
| `intermediate_size` | 1536 |
| `vocab_size` | 32000 |
| `max_position_embeddings` | 1024 |
| Total parameters | ~50M (0.05B) |

---

## Trained Model Details

| Item | Detail |
|------|--------|
| Training steps | 12,000 |
| Corpus size | 19,846 lines |
| Final loss | 4.06 (ppl ≈ 58) |
| Vocabulary | 11,030 tokens (BPE) |
| Optimizer | AdamW (lr=3e-4, cosine w/ warmup) |
| Device | CPU |
| Training time | ~3.2 hours |

---

## Limitations

This is a lightweight educational model. It is **not** suitable for:
- Production deployments
- Medical, legal, or financial applications
- Tasks requiring factual accuracy
- High-quality dialogue or instruction following

The model may produce hallucinated or nonsensical content. Use with appropriate expectations.

---

## Roadmap

- [ ] Expand training corpus to 50K+ lines
- [ ] Add supervised fine-tuning (SFT) with dialogue data
- [ ] Fine-tune vision encoder with image-text pairs (COCO, etc.)
- [ ] Explore larger configs (6 layers / 512d / ~0.1B params)
- [ ] Add more plugins (web search, code execution, etc.)

---

## License

[MIT License](LICENSE) — free for personal and commercial use.

---

## Citation

If you find this project useful, please consider starring the repo or citing:

```bibtex
@misc{stellarai2025,
  author = {AMT Studio},
  title = {StellarAI: A Lightweight Multimodal Language Model},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/amtstudio/StellarAI}
}
```

---

**StellarAI** — Exploring AI, one star at a time.