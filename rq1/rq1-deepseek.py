import sqlite3
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datetime import datetime
from models import *

print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)


model_inputs = [
    ModelInput(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        model_params={"trust_remote_code": True},
        prompt_generation=lambda prompt, text: generate_messages(prompt, text),
        model_creation=lambda input: create_model(input),
        prompts={
            "zero-shot": f"""You are a research assistant that tries to detect disinformation in articles.
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
Do not give any further explanation or justification.  Keep your reasoning output to a minimum.  Be as succinct as possible.  The only relevant output is your final classification.  Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
            "one-shot": f"""You are a research assistant that tries to detect disinformation in articles.
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
Dutch citizens feel the refugees receive preferential treatment.
Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
Do not give any further explanation or justification. Keep your reasoning output to a minimum.  Be as succinct as possible.  The only relevant output is your final classification.  Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
            "few-shot": f"""You are a research assistant that tries to detect disinformation in articles.
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
Here is a first example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
Dutch citizens feel the refugees receive preferential treatment.
Here is a second example of how you can detect disinformation in such articles: war refugees receive a living wage from the government.
Since the number of war refugees is getting higher, the costs of the living wage are increasing.
Many people feel the cost is too high and the living wages are given too easily.  They have to work to get money, so why don't the refugees?
Here is a third example of how you can detect disinformation in such articles: refugees and immigrants often live in difficult circumstances.
They have difficulty understanding the language and culture of the country they are staying in.
Frustrations can in some cases lead to violence.
People generalize this violence, exhibited by a few, to the entire group of refugees.  They perceive all refugees as violent and dangerous, and therefore as not wanted in their country.
Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
Do not give any further explanation or justification. Keep your reasoning output to a minimum.  Be as succinct as possible.  The only relevant output is your final classification.  Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
       }
    )
]


def create_model(model_input: ModelInput):
    tokenizer = AutoTokenizer.from_pretrained(
        model_input.model_name,
        token=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    )
    # Set pad_token_id explicitly if it's not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Set EOS token as pad token

    print(f"""Starting model load in normal mode at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_input.model_name,
        device_map="auto",
        # Use bfloat16 for very large models if supported by A100
        torch_dtype=torch.bfloat16 if "70B" in model_input.model_name else torch.float16,
        token = os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
        **model_input.model_params
    )
    print(f"""Finished model load in normal mode at {datetime.now().strftime("%H:%M:%S")}""", flush=True)
    return tokenizer, model


def process_model(model_input: ModelInput, database: str):

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device {device}", flush=True)

    llm_tokenizer, llm_model = model_input.model_creation(model_input)
    max_tokens = llm_model.config.max_position_embeddings
    print(f"model {model_input.model_name} has max tokens {max_tokens}", flush=True)
    print(f"special tokens map: {llm_tokenizer.special_tokens_map}", flush=True)

    sanitized_model = sanitize_filename(model_input.model_name)
    row_results = {}
    for prompt_type, prompt in model_input.prompts.items():
        progress_file = f"rq1_{sanitized_model}_{prompt_type}.json"

        print(f"processing prompt type {prompt_type} for model {model_input.model_name}", flush=True)
        row_results_for_prompt = []
        existing_row_results_for_prompt = []
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                try:
                    content = f.read()
                    model_result = ModelResult.model_validate_json(content)
                    existing_row_results_for_prompt = model_result.row_results[prompt_type]
                except Exception as e:
                    print(f"could not parse existing file {progress_file}, error {e}", flush=True)
                    existing_row_results_for_prompt = []

        with sqlite3.connect(database) as conn:
            for row in conn.execute(f"select translated_text, disinformation, url from articles"):
                text = row[0]
                text = re.sub(r'\s+', ' ', text)
                ground_truth = 1 if row[1] == 'y' else 0
                url = row[2]

                existing_row_for_url = next((x for x in existing_row_results_for_prompt if x.url == url), None)
                if existing_row_for_url is not None:
                    print(f"skipping url {url} as it is already processed", flush=True)
                    row_results_for_prompt.append(existing_row_for_url)
                    with open(progress_file, "w") as f:
                        f.write(ModelResult(model_input=model_input, row_results={prompt_type: row_results_for_prompt}).model_dump_json(indent=2))
                    continue

                print(f"processing text {text[:100]}", flush=True)

                input_prompt = model_input.prompt_generation(prompt, text)
                try:
                    inputs = llm_tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=max_tokens,
                                           padding=True).to(device)

                    # Attention mask is automatically handled by the tokenizer, but let's confirm it
                    attention_mask = inputs.get("attention_mask", torch.ones(inputs["input_ids"].shape, device=device))

                    # Create attention mask explicitly (if missing)
                    if attention_mask.sum().item() != attention_mask.numel():
                        padding_length = inputs["input_ids"].shape[-1] - attention_mask.sum().item()
                        attention_mask[:, -padding_length:] = 0

                    inputs = {**inputs, "attention_mask": attention_mask.to(device)}

                    # Move inputs to device
                    inputs = {key: value.to(device) for key, value in inputs.items()}

                    result = ''
                    with torch.no_grad():
                        output_ids = llm_model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=1000,
                            temperature=0.1,  # Make output as deterministic as possible
                            num_return_sequences=1,
                            do_sample=False,
                        )
                        result = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

                    if result.startswith(input_prompt):
                        print(f"detected repeated input, skipping", flush=True)
                        result = result[len(input_prompt):]

                    print(f"result of LLM is {result}, ground truth is {row[1]}", flush=True)
                    if not check_result_validity(result):
                        row_results_for_prompt.append(RowResult(url=url, invalid=True, y=ground_truth, y_hat=False, result=result))
                    else:
                        row_results_for_prompt.append(RowResult(url=url, invalid=False, y=ground_truth, y_hat=get_y_hat(result), result=result))
                except Exception as e:
                    row_results_for_prompt.append(
                        RowResult(url=url, invalid=True, y=ground_truth, y_hat=False, result=f"an error occurred : {e}"))

                with open(progress_file, "w") as f:
                    f.write(ModelResult(model_input=model_input, row_results={prompt_type: row_results_for_prompt}).model_dump_json(indent=2))
        row_results[prompt_type] = row_results_for_prompt

    return ModelResult(model_input=model_input, row_results=row_results)


def check_result_validity(result):
    return False

def get_y_hat(result) -> int:
    return -1

def generate_messages_array(prompt: str, text:str):
    # default pipeline kan hier mee om, non default pipeline echter niet
    data = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
    return data

def generate_messages(prompt: str, text:str):
    # default pipeline kan hier mee om, non default pipeline echter niet
    data = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
    return f"{prompt}.  This is the article the user wants to check: {text}.  Your answer to whether this contains disinformation is :"

def generate_messages_mistral(prompt: str, text:str):
    return f"""{prompt}
### Article ###
{text}

### Your answer ### 
    """

def rq1(database: str):
    for model in model_inputs:
        print(f"processing model {model.model_name}", flush=True)
        model_result = process_model(model, database)
        sanitized_model = sanitize_filename(model.model_name)
        with open(f"rq1_{sanitized_model}.json", "w") as f:
            f.write(model_result.model_dump_json(indent=2))

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq1.py <combined-db.sqlite>")
    rq1(sys.argv[1])