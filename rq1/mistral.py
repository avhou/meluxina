from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# deepseek geen probleem,  mistral wel een probleem, llama geen probleem

# llama kent system message, user message en heeft geen assistent nodig
# name = "meta-llama/LLama-3.2-3B-Instruct"


name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
tokenizer = AutoTokenizer.from_pretrained(name)

# Set pad_token_id explicitly if it's not already set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Set EOS token as pad token

model = AutoModelForCausalLM.from_pretrained(
    name,
    device_map="auto",
    torch_dtype=torch.float16,
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    trust_remote_code=True,
)

# Define chat history with separate system and user prompts
# dit is voor llama
# chat = [
#     {"role": "system", "content": "You are an AI assistant and answer only in pirate speak."},
#     {"role": "user", "content": "How are you matey?"},
# ]

# dit is voor mistral
chat = [
    {"role": "system", "content": "You are an AI assistant and answer only in pirate speak."},
    {"role": "user", "content": "How are you matey?"},
]

# Select device (MPS for Mac)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

print(f"chat template is {tokenizer.apply_chat_template(chat, tokenize=False)}")

tokens = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    output_ids = model.generate(
        input_ids=tokens,  # `tokens` is already the correct tensor
        max_new_tokens=100,
        temperature=0.1,  # Make output as deterministic as possible
        num_return_sequences=1,
        do_sample=False,
    )

# Decode output
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\nGenerated Response:\n", response)
