"""

Types of quantization:
    - Affine Quantization. Done.

    - Auto AWQ (Automatic Activation Weight Quantization): This method automatically quantizes the weights and activations of a neural network model.
    - AutoRound (Automatic Rounding Quantization): This method focuses on rounding the weights of a neural network model to a specific bit-width.
    - Bitsand (Bit-Sand Quantization): This method uses a bit-sand approach to quantize the weights and activations of a neural network model.
    - GPTQ (Generalized Post-Training Quantization): This method is a generalized approach to post-training quantization that can be applied to various types of neural network models.

Model being used: TinyLlama-1.1B-Chat-v1.0

"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from datasets import load_dataset
from torch.nn.functional import cross_entropy
import copy
import gc
import numpy as np
import torch.quantization as quantization
import psutil 


"""Load the model and tokenizer from the specified model name."""


def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        device_map=None 
    )
    return tokenizer, model


""" Function for affine quantization of a tensor. """


def quantize_tensor(tensor, num_bits=8):
    qmin, qmax = 0, 2**num_bits - 1
    # Small epsilon to prevent division by zero
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1e-8
    zero_point = torch.round(-min_val / scale).clamp(qmin, qmax)
    quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
    return quantized, scale, zero_point


""" Function to dequantize a quantized tensor back to its original form using the scale and zero point."""


def dequantize_tensor(q_tensor, scale, zero_point):
    return (q_tensor - zero_point) * scale


""" Function to evaluate the perplexity of the model on a given dataset. """


def eval_perplexity(model, tokenizer, dataset, device, n_samples=100):
    """Evaluate perplexity on a dataset with proper error handling and diagnostics"""
    model.eval()
    total_loss = 0.0
    total_length = 0
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            try:
                text = dataset[i]['text']
                
                # Skip empty texts
                if not text.strip():
                    print(f"Skipping empty text at index {i}")
                    continue
                
                # Truncate very long texts to avoid OOM
                if len(text) > 1024:
                    text = text[:1024]
                
                # Print sample info for debugging
                if i == 0:
                    print(f"Sample text: {text[:100]}...")
                
                # Tokenize with proper padding and attention mask
                encodings = tokenizer(text, return_tensors='pt', padding=True, truncation=True, 
                                     max_length=512)
                input_ids = encodings.input_ids.to(device)
                attention_mask = encodings.attention_mask.to(device)
                
                # Skip sequences that are too short
                if input_ids.size(1) <= 1:
                    print(f"Skipping text at index {i} (too short)")
                    continue
                
                # Create shifted labels for causal language modeling
                labels = input_ids.clone()
                
                # Forward pass with proper error handling
                try:
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    # Check for NaN loss
                    if torch.isnan(loss).item():
                        print(f"NaN loss detected at index {i}, skipping")
                        continue
                    
                    # Accumulate loss and token count
                    batch_loss = loss.item() * input_ids.size(1)
                    total_loss += batch_loss
                    total_length += input_ids.size(1)
                    
                    # Print progress
                    if (i+1) % 10 == 0:
                        print(f"Processed {i+1}/{min(n_samples, len(dataset))} samples. "
                              f"Current avg loss: {total_loss/total_length if total_length > 0 else 'N/A'}")
                        
                except RuntimeError as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Unexpected error at index {i}: {e}")
                continue
    
    # Handle edge case where no valid samples were processed
    if total_length == 0:
        print("WARNING: No valid samples were processed. Cannot calculate perplexity.")
        return float('nan')
    
    # Calculate average negative log-likelihood per token
    avg_nll = total_loss / total_length
    
    # Check for NaN or infinity
    if np.isnan(avg_nll) or np.isinf(avg_nll):
        print(f"WARNING: Invalid avg_nll: {avg_nll}. Raw total_loss: {total_loss}, total_length: {total_length}")
        return float('nan')
    
    # Perplexity is exp(average negative log-likelihood)
    try:
        perplexity = np.exp(avg_nll)
        
        # Sanity check on perplexity value
        if perplexity < 1.0:
            print(f"WARNING: Unusually low perplexity: {perplexity}. This may indicate an issue.")
        if perplexity > 10000:
            print(f"WARNING: Unusually high perplexity: {perplexity}. This may indicate an issue.")
            
        return perplexity
    except OverflowError:
        print(f"WARNING: Overflow when calculating exp({avg_nll})")
        return float('nan')



""" Measure the latency of the model for generating text based on a given prompt. """


def measure_latency(model, tokenizer, device, prompt, num_tokens=20, repetitions=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    times = []

    with torch.no_grad():
        for _ in range(repetitions):
            start_time = time.time()
            output = model.generate(**inputs, max_new_tokens=num_tokens)
            end_time = time.time()
            times.append(end_time - start_time)

            # print output
            # decoded_output = tokenizer.decode(
            # output[0], skip_special_tokens=True)
            # print(f"Generated Output: {decoded_output}")

    return times

def quantize_model(model, num_bits=8):
    """Quantize model weights and return a dictionary 
    of quantized weights and their metadata"""
    quantized_state = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Uncommend to skip certain layers
            # if "layernorm" in name.lower() or "embedding" in name.lower():
            #     quantized_state[name] = {"original": param.data.clone()}
            #     continue
                
            q, scale, zp = quantize_tensor(param.data, num_bits)
            quantized_state[name] = {
                "quantized": q,
                "scale": scale,
                "zero_point": zp,
                "original_shape": param.data.shape,
                "original_dtype": param.data.dtype
            }
            
            # For actual memory savings, you need to only keep the quantized values and metadata, not the dequantized values
            
            # Dequantize to test if quantization is correct
            deq = dequantize_tensor(q, scale, zp)
            param.data.copy_(deq)
            
    return quantized_state

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) 

def quantize_model_int8(model):
    """Quantize model using PyTorch's native quantization"""
    # Make a copy of the model for quantization
    model_fp32 = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_fp32.eval()
    
    # Specify quantization configuration
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    model_int8 = torch.quantization.prepare(model_fp32)
    
    # Calibrate with some data (ideally you'd use a calibration dataset)
    # This is a simplified example - in practice, you'd run inference on calibration data
    
    # Convert to quantized model
    model_int8 = torch.quantization.convert(model_int8)
    
    return model_int8

def evaluate_model(model, tokenizer, dataset, device, prompt, n_samples=100, num_tokens=20, repetitions=5):
    """Comprehensive evaluation of model performance with better error handling"""
    results = {}
    
    # 1. Measure memory usage
    start_mem = get_memory_usage()
    
    # 2. Measure model size
    model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Size in MB (assuming float32)
    results['model_size_mb'] = model_size
    
    # 3. Measure perplexity with proper error handling
    print("Calculating perplexity...")
    try:
        model.eval()
        ppl = eval_perplexity(model, tokenizer, dataset, device, n_samples)
        if np.isnan(ppl):
            print("Perplexity calculation failed, trying alternative approach...")
            # Alternative approach: calculate token-by-token loss
            ppl = alternative_perplexity_calculation(model, tokenizer, dataset, device, n_samples=10)
    except Exception as e:
        print(f"Error in perplexity calculation: {e}")
        ppl = float('nan')
    
    results['perplexity'] = ppl
    
    # 4. Measure latency
    print("Measuring latency...")
    try:
        latency_times = measure_latency(model, tokenizer, device, prompt, num_tokens, repetitions)
        results['avg_latency'] = sum(latency_times) / len(latency_times) if latency_times else float('nan')
        results['latency_std'] = np.std(latency_times) if latency_times else float('nan')
        results['latency_p90'] = np.percentile(latency_times, 90) if latency_times else float('nan')
        results['latency_p99'] = np.percentile(latency_times, 99) if latency_times else float('nan')
        results['all_latencies'] = latency_times
    except Exception as e:
        print(f"Error in latency measurement: {e}")
        results['avg_latency'] = float('nan')
        results['latency_std'] = float('nan')
        results['latency_p90'] = float('nan')
        results['latency_p99'] = float('nan')
        results['all_latencies'] = []
    
    # 5. Measure throughput (tokens per second)
    if results['avg_latency'] > 0 and not np.isnan(results['avg_latency']):
        results['tokens_per_second'] = num_tokens / results['avg_latency']
    else:
        results['tokens_per_second'] = float('nan')
    
    # 6. Final memory usage
    end_mem = get_memory_usage()
    results['memory_usage_mb'] = end_mem
    results['memory_increase_mb'] = end_mem - start_mem
    
    return results

def alternative_perplexity_calculation(model, tokenizer, dataset, device, n_samples=10):
    """Alternative approach to calculate perplexity with more diagnostics"""
    model.eval()
    total_log_prob = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            try:
                text = dataset[i]['text']
                if len(text) > 512:
                    text = text[:512]
                
                tokens = tokenizer.encode(text)
                total_tokens += len(tokens) - 1  # Exclude first token for prediction
                
                # Process in smaller chunks to avoid OOM
                for j in range(0, len(tokens)-1, 128):
                    chunk = tokens[j:j+129]  # +1 to include the token to predict
                    if len(chunk) <= 1:
                        continue
                        
                    inputs = torch.tensor([chunk[:-1]]).to(device)
                    targets = torch.tensor([chunk[1:]]).to(device)
                    
                    outputs = model(inputs)
                    logits = outputs.logits
                    
                    # Calculate log probabilities
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # Extract log prob of each target token
                    for k in range(len(targets[0])):
                        total_log_prob -= log_probs[0, k, targets[0][k]].item()
                
                if (i+1) % 2 == 0:
                    print(f"Alternative method: Processed {i+1}/{min(n_samples, len(dataset))} samples")
                    
            except Exception as e:
                print(f"Error in alternative perplexity calculation at sample {i}: {e}")
                continue
    
    if total_tokens == 0:
        return float('nan')
        
    # Calculate perplexity
    avg_neg_log_prob = total_log_prob / total_tokens
    return np.exp(avg_neg_log_prob)

def visualize_comparison(original_results, quantized_results):
    """Create comprehensive visualizations comparing original and quantized models"""
    # 1. Create a comparison table for all metrics
    metrics = ['model_size_mb', 'perplexity', 'avg_latency', 'tokens_per_second', 'memory_usage_mb']
    labels = ['Model Size (MB)', 'Perplexity', 'Avg Latency (s)', 'Tokens/Second', 'Memory Usage (MB)']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [original_results[metric], quantized_results[metric]]
        model_types = ['Original', 'Quantized']
        
        sns.barplot(x=model_types, y=values, palette='viridis', ax=axes[i])
        axes[i].set_title(f'{label} Comparison')
        axes[i].set_ylabel(label)
        
        # Add percentage change
        if original_results[metric] > 0:
            pct_change = (quantized_results[metric] - original_results[metric]) / original_results[metric] * 100
            direction = "increase" if pct_change > 0 else "decrease"
            axes[i].text(1, values[1], f"{abs(pct_change):.1f}% {direction}", 
                         ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png')
    
    # 2. Latency distribution plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(original_results['all_latencies'], label='Original Model', shade=True)
    sns.kdeplot(quantized_results['all_latencies'], label='Quantized Model', shade=True)
    plt.title('Latency Distribution: Original vs Quantized Model')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('latency_distribution.png')

if __name__ == '__main__':
    # Load the model and tokenizer
    start_time = time.time()

    tokenizer, model = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    end_time = time.time()
    print("Model loaded in {:.2f} seconds".format(end_time - start_time))

    # Load dataset for ppl calculations
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Define evaluation prompt
    prompt = "Once upon a time in a land far away"
    
    # Evaluate original model
    print(colored("Evaluating original model...", "blue"))
    original_results = evaluate_model(model, tokenizer, dataset, device, prompt)
    original_results['all_latencies'] = measure_latency(
        model, tokenizer, device, prompt, num_tokens=25, repetitions=50)
    
    print(colored("Original model evaluation:", "green"))
    for key, value in original_results.items():
        if key != 'all_latencies':
            print(f"  {key}: {value}")
    
    # Quantize the model using PyTorch's native INT8 quantization
    print(colored("\nQuantizing model to INT8...", "blue"))
    start_time = time.time()
    
    # First approach: Custom affine quantization
    quantized_state = quantize_model(model, num_bits=8)
    
    # Second approach: PyTorch native quantization (uncomment to use)
    # Note: This may not work directly with all transformer models without modifications
    # model_int8 = quantize_model_int8(model)
    # model = model_int8
    
    end_time = time.time()
    print("Model quantization completed in {:.2f} seconds".format(
        end_time - start_time))
    
    # Evaluate quantized model
    print(colored("\nEvaluating quantized model...", "blue"))
    quantized_results = evaluate_model(model, tokenizer, dataset, device, prompt)
    quantized_results['all_latencies'] = measure_latency(
        model, tokenizer, device, prompt, num_tokens=25, repetitions=50)
    
    print(colored("Quantized model evaluation:", "green"))
    for key, value in quantized_results.items():
        if key != 'all_latencies':
            print(f"  {key}: {value}")
    
    # Create visualizations comparing the models
    print(colored("\nGenerating comparison visualizations...", "blue"))
    visualize_comparison(original_results, quantized_results)
    
    # Clean up memory
    del model
    del tokenizer
    gc.collect()
    
    print(colored("\nEvaluation complete. Visualizations saved.", "green"))
