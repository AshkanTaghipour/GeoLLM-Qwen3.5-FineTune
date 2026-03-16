# Quantizing GeoLLM-Qwen3.5-27B for Local Deployment

Guide for converting the fine-tuned 27B model to GGUF format and running it locally via **Ollama** or **llama-server**.

---

## Recommended Quantizations

For a domain-specific fine-tune, prefer higher-quality quants to preserve the geological knowledge trained into the model.

| Quant | File Size | VRAM Needed | Perplexity Loss | Use Case |
|:------|:----------|:------------|:----------------|:---------|
| **Q5_K_M** | ~19 GB | ~22 GB | +0.014 (very low) | **Primary recommendation** — best quality/size tradeoff |
| **Q4_K_M** | ~17 GB | ~20 GB | +0.054 (low) | **Budget option** — fits 24 GB GPUs, Ollama's default for 27B |
| Q6_K | ~22 GB | ~25 GB | +0.004 (negligible) | Near-lossless, if you have the VRAM |
| Q8_0 | ~28 GB | ~32 GB | +0.0004 (zero) | Virtually lossless, large file |

Perplexity figures are relative to unquantized, from llama.cpp benchmarks on 7B models — directionally accurate for 27B.

**Avoid Q3_K and below** — perplexity degradation is steep and risks losing fine-tuned domain knowledge.

### Unsloth Dynamic 2.0

Unsloth's Dynamic 2.0 quantization selectively adjusts precision per-layer rather than applying a uniform quant. It outperforms standard K-quants and imatrix on benchmarks. If available for your model, prefer Dynamic 2.0 GGUFs over standard quants.

---

## Conversion Pipeline

### Prerequisites

```bash
# Clone and build llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON    # or -DGGML_METAL=ON for Mac
cmake --build build --config Release -j

# Install Python dependencies for conversion
pip install -r requirements.txt
```

### Step 1: Download the Merged Model

```bash
huggingface-cli download AshkanTaghipour/GeoLLM-Qwen3.5-27B --local-dir ./GeoLLM-27B
```

This downloads the merged bf16 safetensors (~55 GB).

### Step 2: Convert to F16 GGUF

```bash
python convert_hf_to_gguf.py ./GeoLLM-27B --outtype f16 --outfile GeoLLM-27B-f16.gguf
```

### Step 3: Quantize

```bash
./build/bin/llama-quantize GeoLLM-27B-f16.gguf GeoLLM-27B-Q5_K_M.gguf Q5_K_M
./build/bin/llama-quantize GeoLLM-27B-f16.gguf GeoLLM-27B-Q4_K_M.gguf Q4_K_M
```

You can delete the F16 GGUF after quantization to reclaim disk space.

---

## Running with Ollama

### Option A: Import a Pre-built GGUF

```bash
cat > Modelfile <<'EOF'
FROM ./GeoLLM-27B-Q5_K_M.gguf
SYSTEM "You are a specialist geologist with expertise in Western Australian mineral exploration."
PARAMETER temperature 0.6
PARAMETER top_p 0.95
EOF

ollama create geollm-27b -f Modelfile
ollama run geollm-27b
```

### Option B: Let Ollama Quantize from Safetensors

```bash
cat > Modelfile <<'EOF'
FROM ./GeoLLM-27B
SYSTEM "You are a specialist geologist with expertise in Western Australian mineral exploration."
PARAMETER temperature 0.6
PARAMETER top_p 0.95
EOF

ollama create geollm-27b-q5km -f Modelfile -q Q5_K_M
```

---

## Running with llama-server

```bash
./build/bin/llama-server \
  -m GeoLLM-27B-Q5_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 4096
```

This exposes an OpenAI-compatible API at `http://localhost:8080`. Key flags:

- `-ngl 99` — offload all layers to GPU
- `-c 4096` — context window (increase if needed, costs more VRAM)

### Verify It Works

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a specialist geologist with expertise in Western Australian mineral exploration."},
      {"role": "user", "content": "What geophysical methods target komatiite nickel in the Eastern Goldfields?"}
    ],
    "temperature": 0.6
  }'
```

---

## Memory Requirements

### What Actually Uses Memory

VRAM usage is **not** just the GGUF file size. Total memory = model weights + KV cache + compute buffers + OS overhead.

| Component | Description |
|:----------|:------------|
| **Model weights** | The GGUF file loaded into memory (≈ file size) |
| **KV cache** | Grows linearly with context length — often the hidden killer |
| **Compute buffers** | Scratch space for intermediate activations (~500 MB–1 GB) |
| **OS/driver overhead** | CUDA/Metal runtime, ~500 MB–1 GB |

### KV Cache for Qwen3.5-27B

Qwen3.5-27B uses Grouped-Query Attention (GQA) with 8 KV heads, 64 layers, and head dimension 128. The KV cache formula:

```
KV cache (bytes) = 2 × num_layers × num_kv_heads × head_dim × context_length × bytes_per_element
```

At FP16 (2 bytes per element):

| Context Length | KV Cache Size |
|:---------------|:--------------|
| 2,048 tokens | ~0.5 GB |
| 4,096 tokens | ~1.0 GB |
| 8,192 tokens | ~2.0 GB |
| 16,384 tokens | ~4.0 GB |
| 32,768 tokens | ~8.0 GB |

Ollama supports KV cache quantization (Q8/Q4) which can halve or quarter these numbers.

### Total VRAM Estimates (27B Model)

Includes weights + KV cache at 4K context + ~1.5 GB overhead:

| Quant | Weights | + KV (4K ctx) | + Overhead | **Total** |
|:------|:--------|:--------------|:-----------|:----------|
| Q4_K_M | ~17 GB | ~1.0 GB | ~1.5 GB | **~19.5 GB** |
| Q5_K_M | ~19 GB | ~1.0 GB | ~1.5 GB | **~21.5 GB** |
| Q6_K | ~22 GB | ~1.0 GB | ~1.5 GB | **~24.5 GB** |
| Q8_0 | ~28 GB | ~1.0 GB | ~1.5 GB | **~30.5 GB** |

At 8K context, add another ~1 GB. At 32K context, add ~7 GB more.

### Hardware Compatibility

| Hardware | VRAM/RAM | Best Quant | Max Context | Notes |
|:---------|:---------|:-----------|:------------|:------|
| RTX 4090 / 5090 (24 GB) | 24 GB | Q4_K_M | ~8K | Tight — use KV cache quantization for longer context |
| RTX A5000 / A6000 (48 GB) | 48 GB | Q5_K_M or Q6_K | 32K+ | Comfortable headroom |
| A100 (80 GB) | 80 GB | Q8_0 or Q6_K | 32K+ | No constraints |
| Apple M2/M3 Pro (32 GB unified) | 32 GB | Q4_K_M | ~8K | Shared with OS — leave ~6 GB free |
| Apple M2/M3 Max (64 GB unified) | 64 GB | Q5_K_M or Q6_K | 32K+ | Excellent for local dev |
| CPU-only (64 GB RAM) | 64 GB | Q4_K_M | 4K | Works but slow (~2–5 tok/s) |

### Important Caveats

- **16 GB GPUs** (RTX 4080, 5060 Ti): Q4_K_M is ~17 GB for weights alone. Real-world reports confirm it **fails to load** even at reduced context. Use CPU offloading (`-ngl` partial) or a smaller model.
- **Context is the hidden cost**: Doubling context length roughly doubles KV cache. A 32K context window on Q4_K_M pushes total VRAM to ~27 GB.
- **Ollama KV quantization**: Ollama supports `--kv-cache-type q8_0` or `q4_0` which significantly reduces KV cache memory, enabling longer contexts on constrained hardware.
- **Partial GPU offload**: Both llama-server and Ollama support offloading only some layers to GPU (`-ngl 40` instead of `99`), keeping the rest in system RAM. Slower but fits larger models.

---

## References

- [llama.cpp quantization types and perplexity](https://github.com/ggml-org/llama.cpp/discussions/2094)
- [Ollama GGUF import docs](https://github.com/ollama/ollama/blob/main/docs/import.md)
- [Unsloth Dynamic 2.0 GGUFs](https://docs.unsloth.ai/basics/dynamic-v2.0)
- [GGUF quantization quality vs speed on consumer GPUs](https://dasroot.net/posts/2026/02/gguf-quantization-quality-speed-consumer-gpus/)
