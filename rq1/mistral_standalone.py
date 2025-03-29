import transformers
import torch
from datetime import datetime

from transformers import AutoTokenizer

model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

print(f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="balanced_low_0",
)

print(f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
