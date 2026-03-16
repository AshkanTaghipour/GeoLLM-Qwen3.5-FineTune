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

## VRAM Reality Check

The Q4_K_M quant is ~17 GB on disk, but actual VRAM usage is higher due to KV cache and context window overhead. Expect:

- **16 GB GPU**: May fail to load Q4_K_M with default context. Reduce `-c` to 2048 or use CPU offloading.
- **24 GB GPU**: Runs Q4_K_M and Q5_K_M comfortably at 4K context.
- **32+ GB GPU**: Runs Q6_K or Q8_0 without issues.
- **Apple Silicon (32 GB unified)**: Q4_K_M and Q5_K_M work well via Metal.

---

## References

- [llama.cpp quantization types and perplexity](https://github.com/ggml-org/llama.cpp/discussions/2094)
- [Ollama GGUF import docs](https://github.com/ollama/ollama/blob/main/docs/import.md)
- [Unsloth Dynamic 2.0 GGUFs](https://docs.unsloth.ai/basics/dynamic-v2.0)
- [GGUF quantization quality vs speed on consumer GPUs](https://dasroot.net/posts/2026/02/gguf-quantization-quality-speed-consumer-gpus/)
