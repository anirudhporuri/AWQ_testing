import os
import time
import torch
import psutil
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tabulate import tabulate

# Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#model_id = "meta-llama/Llama-2-7b-hf"
#prompt = "Once upon a time in a faraway kingdom,"
prompt = "In a groundbreaking experiment in quantum physics,"
eval_file = "tiny_eval.txt"

# Benchmarking helpers
def benchmark_latency(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end = time.time()
    return end - start

def compute_perplexity_from_file(model, tokenizer, filepath, device):
    with open(filepath, 'r') as f:
        text = f.read()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

def get_gpu_memory_usage_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def estimate_param_size(model, bytes_per_param):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * bytes_per_param / (1024 ** 2)  # MB

def benchmark_model(model_name, model, tokenizer, device, bytes_per_param):
    latency = benchmark_latency(model, tokenizer, prompt)
    mem_usage = get_gpu_memory_usage_mb()
    param_size = estimate_param_size(model, bytes_per_param)
    output_text = generate_text(model, tokenizer, prompt)
    ppl = compute_perplexity_from_file(model, tokenizer, eval_file, device)
    return [model_name, f"{latency:.2f}", f"{mem_usage:.2f}", f"{param_size:.2f}", f"{ppl:.2f}", output_text[:100] + "..."]

# ---------------------
# Load original FP16 model
print("Loading original (FP16) model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
original_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, revision="main")
original_results = benchmark_model("Original (FP16)", original_model, tokenizer, original_model.device, 2)  # 2 bytes for FP16

# Load quantized 8-bit model
print("Loading quantized (8-bit BnB) model...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False #True <- this causes memory and latency to be higher than the og model bc this offloads to cpu, setting to false should solve this
)

quant_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    revision="main"
)
quant_results = benchmark_model("Quantized (8-bit)", quant_model, tokenizer, quant_model.device, 1)  # 1 byte for INT8
torch.cuda.reset_peak_memory_stats()

# --- AWQ Benchmarking Branch ---

from quantization import (
    collect_activation_stats,
    awq_quantize_model,
    load_awq_model_from_state,
    load_model  # if not already imported
)

# 1) Calibration on a handful of prompts
cal_texts = [
    "In a groundbreaking experiment in quantum physics,",
    "Once upon a time in a faraway kingdom,",
    # …add 10–20 varied calibration snippets…
]
print("Calibrating activations for AWQ...")
stats = collect_activation_stats(
    original_model, tokenizer,
    cal_texts, original_model.device
)

# 2) Perform AWQ (4‑bit) quantization in blocks of 128×32
print("Quantizing model with AWQ (4-bit)...")
awq_state = awq_quantize_model(
    original_model, stats,
    num_bits=4,
    block_size=(128, 32)
)

# 3) Load a fresh copy and inject the AWQ‑quantized weights
print("Loading fresh model and applying AWQ weights...")
tokenizer_awq, awq_model = load_model(model_id)
awq_model = load_awq_model_from_state(awq_model, awq_state)

# 4) Benchmark the AWQ‑quantized model
awq_results = benchmark_model(
    "Quantized (AWQ 4-bit)",
    awq_model, tokenizer_awq,
    awq_model.device,
    bytes_per_param=0.5
)

# 5) Add AWQ results to your table and CSV
table = [original_results, quant_results, awq_results]
print("\n=== Benchmark Comparison (incl. AWQ) ===")
print(tabulate(table, headers=headers))

with open("benchmark_results.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["--- AWQ 4-bit Results ---"])
    writer.writerow(headers)
    writer.writerow(awq_results)

# ---------------------
# Print table
headers = ["Model", "Latency (s)", "Memory (MB)", "Param Size (MB)", "Perplexity", "Sample Output"]
table = [original_results, quant_results]

print("\n=== Benchmark Comparison ===")
print(tabulate(table, headers=headers))

# ---------------------
# Save to CSV
with open("benchmark_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows([original_results, quant_results])

print("\nResults saved to benchmark_results.csv")
