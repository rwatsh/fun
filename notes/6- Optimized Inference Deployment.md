- Frameworks for optimizing LLM deployments
	- Text Generation Inference (TGI)
	- vLLM
	- llama.cpp
- **TGI** is designed to be stable and predictable in production, using fixed sequence lengths to keep memory usage consistent. TGI manages memory using Flash Attention 2 and continuous batching techniques. This means it can process attention calculations very efficiently and keep the GPU busy by constantly feeding it work.
	- Flash Attention is a technique that optimizes the attention mechanism in transformer models by addressing memory bandwidth bottlenecks.
	-  the attention mechanism has quadratic complexity and memory usage, making it inefficient for long sequences.
	- The key innovation is in how it manages memory transfers between High Bandwidth Memory (HBM) and faster SRAM cache. Traditional attention repeatedly transfers data between HBM and SRAM, creating bottlenecks by leaving the GPU idle. Flash Attention loads data once into SRAM and performs all calculations there, minimizing expensive memory transfers.
	- While the benefits are most significant during training, Flash Attention’s reduced VRAM usage and improved efficiency make it valuable for inference as well, enabling faster and more scalable LLM serving.
```bash
docker run --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct
# Failed on my machine with:
# docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

```python
from openai import OpenAI

# Initialize client pointing to TGI endpoint
client = OpenAI(
    base_url="http://localhost:8080/v1",  # Make sure to include /v1
    api_key="not-needed",  # TGI doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```
- **vLLM** takes a different approach by using PagedAttention. Just like how a computer manages its memory in pages, vLLM splits the model’s memory into smaller blocks. This clever system means it can handle different-sized requests more flexibly and doesn’t waste memory space. It’s particularly good at sharing memory between different requests and reduces memory fragmentation, which makes the whole system more efficient.
	- PagedAttention is a technique that addresses another critical bottleneck in LLM inference: KV cache memory management.
	- The PagedAttention approach can lead to up to 24x higher throughput compared to traditional methods, making it a game-changer for production LLM deployments.
	- https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

```python
from openai import OpenAI

# Initialize client pointing to vLLM endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM doesn't require an API key by default
)

# Chat completion
response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)
```

- **llama.cpp** is a highly optimized C/C++ implementation originally designed for running LLaMA models on consumer hardware. It focuses on CPU efficiency with optional GPU acceleration and is ideal for resource-constrained environments.
	- Quantization in llama.cpp reduces the precision of model weights from 32-bit or 16-bit floating point to lower precision formats like 8-bit integers (INT8), 4-bit, or even lower. This significantly reduces memory usage and improves inference speed with minimal quality loss.
	- This approach enables running billion-parameter models on consumer hardware with limited memory, making it perfect for local deployments and edge devices.
	- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
	- https://github.com/ggml-org/llama.cpp
	- https://github.com/abetlen/llama-cpp-python
```bash
# Clone the repository
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp


sudo apt update &&
sudo apt upgrade &&
sudo apt install curl libcurl4-openssl-dev

cmake -B build
cmake --build build --config Release

cd build/bin

# Start the server
./llama-server \
-hf HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M \
--host 0.0.0.0 \
--port 8080 \
-c 4096 \
--threads 8 \
--batch-size 512 \
--n-gpu-layers 0     # GPU layers (0 = CPU only)
```

```python
from openai import OpenAI

# Initialize client pointing to llama.cpp server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",  # llama.cpp server requires this placeholder
)

# Chat completion
response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",  # Model identifier can be anything as server only loads one model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Advanced parameters example

response = client.chat.completions.create(
model="smollm2-1.7b-instruct",
messages=[
{"role": "system", "content": "You are a creative storyteller."},
{"role": "user", "content": "Write a creative story"},
],
temperature=0.8, # Higher for more creativity
top_p=0.95, # Nucleus sampling probability
frequency_penalty=0.5, # Reduce repetition of frequent tokens
presence_penalty=0.5, # Reduce repetition by penalizing tokens already present
max_tokens=100, # Maximum generation length
)

print(response.choices[0].message.content)
```
- 