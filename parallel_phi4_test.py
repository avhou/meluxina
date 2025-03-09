import transformers
from datetime import datetime
import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

start = datetime.now()
current_time = start.strftime("%H:%M:%S")
print(f"Starting at {current_time}", flush=True)

print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)



# Initialize accelerator
accelerator = Accelerator()

# Load model and tokenizer
model_name = "microsoft/phi-4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Dispatch model across all GPUs
model = accelerator.prepare(model)

# Create pipeline with parallel execution
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="balanced")

system_prompt = "You are an expert AI system that specializes in named entity recognition and knowledge graph extraction.  You are designed take in any input text, extract the relevant information, and output a knowledge graph in turtle (TTL) format.  Do not provide any explanation or justification.  Only output TTL knowledge graphs."
user_prompt = """
Vlaams Belang strongly disagrees with the purple-Green government planning billions of euros without any discussion on living wages for refugees Ukrainians.
About 665 million euros until the end of July.
This is simply priceless debt, says representative Wouter vermeersch.
Paid-up benefit is not required for this purpose.
This can be done by providing, according to Dutch example, in bed, bath and bread.
Since Vermeersch's questions, the Secretary of State for Budget Eva De Bleeker suggested that he should assume a total influx of 259 000 refugees until summer.
Of these 20 percent would need shelter.
About 49 million euros of cash benefits are calculated until the end of July, says Vermeersch.
In the coming months, the influx is gradually being built up.
De Bleeker estimates the cost of living wages at 665 million euros until the end of July.
259.000 Ukrainians cost 259 million euro per month or 3.1 billion euros per year in living wages alone, which is unaffordable "If we give 200,000 Ukrainian refugees an average of 1,000 euros a month, this costs 200 million euro/month and therefore 2.4 billion euro per year.
That's just not possible for public finances. It will be paid in due course by all working people in our country.
Unfortunately, De Bleeker does not even take a message here, she simply does not go into it.
• If some of these figures are confirmed here, such as the figure of the expected 259 000 refugees, then the figure calculated by the party members of the State Secretary of State is 2.4 billion in living wages per year very realistic: Vermeersch decides.
♪Flighted Ukrainians must get safe accommodation, as well as the necessary medical and social assistance.
Living wages are unnecessary.
"""

user_texts = [user_prompt] * 10

messages = [
    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    for user_text in user_texts
]


print(f"start executing the pipeline", flush=True)

# Parallel execution
outputs = accelerator.gather(pipeline(messages))

# Print results
for output in outputs:
    print(f"{output[0]['generated_text']}\n")

stop = datetime.now()
current_time = stop.strftime("%H:%M:%S")
print(f"Stopping at {current_time}", flush=True)


