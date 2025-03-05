import os
import requests
from transformers import AutoModel, AutoConfig

# ✅ Ensure Hugging Face token is available
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN is not set. Please export it in your environment.")

def get_model_metadata(model_name):
    """Fetch model metadata using the Hugging Face API."""
    api_url = f"https://huggingface.co/api/models/{model_name}"
    
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    response = requests.get(api_url, headers=headers)

    if response.status_code == 401:
        raise ValueError("Unauthorized. Ensure your Hugging Face token has access.")

    if response.status_code == 404:
        raise ValueError("Model not found. Check if the repository exists and is public.")

    if response.status_code != 200:
        raise ValueError(f"Error fetching metadata: {response.status_code} - {response.text}")

    return response.json()

def get_model_info(model_name):
    """Load model config and compute resource requirements."""
    config = AutoConfig.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model", None)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "num_layers", None))
    seq_length = getattr(config, "max_position_embeddings", 4096)  # Default to 4096

    # Load model without gradients
    model = AutoModel.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
    
    # Total parameters in billions
    total_params = sum(p.numel() for p in model.parameters()) / 1e9  # Convert to billions

    return total_params, num_layers, hidden_size, seq_length

def estimate_memory_requirements(total_params, hidden_size, num_layers, quantization="float16"):
    """Estimate memory requirements based on model parameters and quantization."""
    dtype_size = {"fp32": 4, "float16": 2, "int8": 1, "int4": 0.5}.get(quantization, 2)
    
    # Memory for weights
    weight_memory_gb = (total_params * dtype_size) / (1024**3)

    # KV Cache estimation per token
    kv_cache_per_token_mb = (hidden_size * num_layers * 2 * dtype_size) / (1024**2)

    # Total KV Cache memory (assuming 4096 sequence length)
    total_kv_cache_gb = (kv_cache_per_token_mb * 4096) / 1024  

    # Activation and buffer memory (rough estimate, depends on batch size)
    activation_memory_gb = total_kv_cache_gb * 0.5  

    # Total memory required
    total_memory_gb = weight_memory_gb + total_kv_cache_gb + activation_memory_gb + 2  # Adding overhead

    return weight_memory_gb, kv_cache_per_token_mb, total_kv_cache_gb, activation_memory_gb, total_memory_gb

def main(model_name):
    print(f"Fetching model details for: {model_name}")

    metadata = get_model_metadata(model_name)
    print(f"Model Name: {model_name}")

    total_params, num_layers, hidden_size, seq_length = get_model_info(model_name)
    
    print(f"Total Parameters: {total_params:.2f}B")
    print(f"Layers (L): {num_layers}")
    print(f"Hidden Size (H): {hidden_size}")
    print(f"Sequence Length (S): {seq_length}")

    weight_memory_gb, kv_cache_per_token_mb, total_kv_cache_gb, activation_memory_gb, total_memory_gb = estimate_memory_requirements(
        total_params, hidden_size, num_layers
    )

    print(f"Memory for Weights: {weight_memory_gb:.2f} GB")
    print(f"KV Cache per Token: {kv_cache_per_token_mb:.2f} MB")
    print(f"Total KV Cache Memory: {total_kv_cache_gb:.2f} GB")
    print(f"Activations & Buffers: {activation_memory_gb:.2f} GB")
    print(f"Total Memory Required: {total_memory_gb:.2f} GB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Hugging Face model metadata.")
    parser.add_argument("model_name", type=str, help="Hugging Face model name (e.g., microsoft/Phi-3.5-mini-instruct)")
    args = parser.parse_args()

    main(args.model_name)