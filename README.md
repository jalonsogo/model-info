🚀 Hugging Face Model Info Extractor

📌 Overview

The Hugging Face Model Info Extractor is a Python script that retrieves detailed metadata and memory requirements for a given model from Hugging Face.

With this script, you can:

Fetch model details such as total parameters, layers, hidden size, and sequence length.

Estimate memory consumption (weights, KV cache, activations, total memory).

Handle private and gated models with authentication.

Work with different quantization types (FP32, FP16, INT8, INT4).

📥 Installation

Ensure you have Python installed, then install the necessary dependencies:

pip install transformers torch requests

🚀 Usage

To retrieve model metadata, simply run:

python hf_model_info.py microsoft/Phi-3.5-mini-instruct

You can also pass the full Hugging Face model URL:

python hf_model_info.py https://huggingface.co/microsoft/Phi-3.5-mini-instruct

If the model is private or gated, set your Hugging Face token:

export HUGGINGFACE_TOKEN=your_token_here  # macOS/Linux
set HUGGINGFACE_TOKEN=your_token_here     # Windows CMD
$env:HUGGINGFACE_TOKEN="your_token_here"  # Windows PowerShell

📊 Example Output

Fetching model details for: microsoft/Phi-3.5-mini-instruct
Model Name: microsoft/Phi-3.5-mini-instruct
Total Parameters: 3.80B
Layers (L): 32
Hidden Size (H): 4096
Sequence Length (S): 4096
Memory for Weights: 7.60 GB
KV Cache per Token: 2.00 MB
Total KV Cache Memory: 8.00 GB
Activations & Buffers: 4.00 GB
Total Memory Required: 21.60 GB

🔥 Features

✅ Supports both public and private models
✅ Handles authentication for gated models
✅ Computes model size and memory usage
✅ Supports different quantization types
✅ Works with text and vision models


📜 LOLcense

For {root} sake I'm a designer. Mostly all the code has been writen by chatGPT and ad latere.