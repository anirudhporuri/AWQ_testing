from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained(
    "/Users/mukunds/.llama/checkpoints/Llama3.2-1B")
print(f"Vocab size: {tokenizer.vocab_size}")
