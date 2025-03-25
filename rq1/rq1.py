import sqlite3
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Literal, Any, Optional, Callable
from pydantic import BaseModel, Field
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
from accelerate import infer_auto_device_map, dispatch_model
from datetime import datetime

print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)

PromptType = Literal["zero-shot", "one-shot", "few-shot"]

class Output(BaseModel):
    contains_disinformation: bool

class ModelInput(BaseModel):
    model_name: str
    model_params: Dict[str, Any]
    prompts: Dict[PromptType, str]
    prompt_generation: Callable[[str, str], str] = Field(default_factory=lambda: lambda prompt, text: "", exclude=True)
    model_creation: Callable[[Any], Any] = Field(default_factory=lambda: lambda input: None, exclude=True)


# model_inputs = [
#     ModelInput(
#         model_name="meta-llama/Llama-3.2-3B-Instruct",
#         model_params={},
#         prompt_generation=lambda prompt, text: generate_messages(prompt, text),
#         model_creation=lambda input: create_model(input),
#         prompts={
#             "zero-shot": f"""You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#             "one-shot": """You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
# They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
# Dutch citizens feel the refugees receive preferential treatment.
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#             "few-shot": """You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Here is a first example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
# They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
# Dutch citizens feel the refugees receive preferential treatment.
# Here is a second example of how you can detect disinformation in such articles: war refugees receive a living wage from the government.
# Since the number of war refugees is getting higher, the costs of the living wage are increasing.
# Many people feel the cost is too high and the living wages are given too easily.  They have to work to get money, so why don't the refugees?
# Here is a third example of how you can detect disinformation in such articles: refugees and immigrants often live in difficult circumstances.
# They have difficulty understanding the language and culture of the country they are staying in.
# Frustrations can in some cases lead to violence.
# People generalize this violence, exhibited by a few, to the entire group of refugees.  They perceive all refugees as violent and dangerous, and therefore as not wanted in their country.
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#        }
#     ),
# ]

model_inputs = [
#     ModelInput(
#         model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
#         model_params={"trust_remote_code": True},
#         prompt_generation=lambda prompt, text: generate_messages(prompt, text),
#         model_creation=lambda input: create_model(input),
#         prompts={
#             "zero-shot": f"""You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#             "one-shot": f"""You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
# They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
# Dutch citizens feel the refugees receive preferential treatment.
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#             "few-shot": f"""You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Here is a first example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
# They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
# Dutch citizens feel the refugees receive preferential treatment.
# Here is a second example of how you can detect disinformation in such articles: war refugees receive a living wage from the government.
# Since the number of war refugees is getting higher, the costs of the living wage are increasing.
# Many people feel the cost is too high and the living wages are given too easily.  They have to work to get money, so why don't the refugees?
# Here is a third example of how you can detect disinformation in such articles: refugees and immigrants often live in difficult circumstances.
# They have difficulty understanding the language and culture of the country they are staying in.
# Frustrations can in some cases lead to violence.
# People generalize this violence, exhibited by a few, to the entire group of refugees.  They perceive all refugees as violent and dangerous, and therefore as not wanted in their country.
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#        }
#     ),
#     ModelInput(
#         model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
#         model_params={"trust_remote_code": True, "offload_folder": "offload", "low_cpu_mem_usage": True},
#         prompt_generation=lambda prompt, text: generate_messages_mistral(prompt, text),
#         model_creation=lambda input: create_model(input),
#         prompts={
#             "zero-shot": f"""### Instruction ###
# You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification.
#
# ### Output format ###
# Respond with a valid JSON object, following this schema:
# {Output.model_json_schema()}
# """,
#             "one-shot": f"""You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
# They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
# Dutch citizens feel the refugees receive preferential treatment.
# Answer with a simple yes or no, yes if you think the article contains disinformation, no if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#             "few-shot": f"""You are a research assistant that tries to detect disinformation in articles.
# A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
# (that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
# Here is a first example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
# They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
# Dutch citizens feel the refugees receive preferential treatment.
# Here is a second example of how you can detect disinformation in such articles: war refugees receive a living wage from the government.
# Since the number of war refugees is getting higher, the costs of the living wage are increasing.
# Many people feel the cost is too high and the living wages are given too easily.  They have to work to get money, so why don't the refugees?
# Here is a third example of how you can detect disinformation in such articles: refugees and immigrants often live in difficult circumstances.
# They have difficulty understanding the language and culture of the country they are staying in.
# Frustrations can in some cases lead to violence.
# People generalize this violence, exhibited by a few, to the entire group of refugees.  They perceive all refugees as violent and dangerous, and therefore as not wanted in their country.
# Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
# Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
#        }
#     ),
    ModelInput(
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        model_params={"trust_remote_code": True},
        prompt_generation=lambda prompt, text: generate_messages(prompt, text),
        model_creation=lambda input: create_model_llama(input),
        prompts={
            "zero-shot": f"""You are a research assistant that tries to detect disinformation in articles.
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
            "one-shot": f"""You are a research assistant that tries to detect disinformation in articles.
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
Here is one example of how you can detect disinformation in such articles: war refugees flee their country and go to the Netherlands.
They can immediately stay in the social housing infrastructure, whereas Dutch citizens may have to wait for many years to become eligible for social housing.
Dutch citizens feel the refugees receive preferential treatment.
Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
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
Do not give any further explanation or justification. Generate your output in JSON format.  The output should conform to this JSON schema : {Output.model_json_schema()}.""",
       }
    ),
]


class RowResult(BaseModel):
    url: str
    invalid: bool
    result: str
    y: int
    y_hat: int

class ModelResult(BaseModel):
    model_input: ModelInput
    row_results: Dict[PromptType, List[RowResult]]

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


def create_model_llama(model_input: ModelInput):
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
        device_map="balanced_low_0",
        # Use bfloat16 for very large models if supported by A100
        torch_dtype=torch.bfloat16 if "70B" in model_input.model_name else torch.float16,
        token = os.environ.get('HUGGINGFACEHUB_API_TOKEN'),
        **model_input.model_params
    )

    print(f"""Finished dispatch model at {datetime.now().strftime("%H:%M:%S")}""", flush=True)
    return tokenizer, model


def create_model_mistral(model_input: ModelInput):
    tokenizer = AutoTokenizer.from_pretrained(
        model_input.model_name,
        token=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    )
    # Set pad_token_id explicitly if it's not already set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Set EOS token as pad token

    print(f"""Starting model load in meta mode at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_input.model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="meta"
    )
    print(f"""Finished model load in meta mode at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    device_map = infer_auto_device_map(model, max_memory={0: "40GB", 1: "40GB", 2: "40GB", 3: "40GB"},
                                       no_split_module_classes=["MistralDecoderLayer"])

    print(f"""Starting model load in normal mode at {datetime.now().strftime("%H:%M:%S")}""", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_input.model_name,
        device_map=device_map,
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
            for row in conn.execute(f"select translated_text, disinformation, url from articles limit 2"):
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
                inputs = llm_tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=max_tokens, padding=True).to(device)

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
                        max_new_tokens=500,
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

                with open(progress_file, "w") as f:
                    f.write(ModelResult(model_input=model_input, row_results={prompt_type: row_results_for_prompt}).model_dump_json(indent=2))
        row_results[prompt_type] = row_results_for_prompt

    return ModelResult(model_input=model_input, row_results=row_results)


def clean_result(result: str) -> str:
    cleaned_result = re.sub(r'```json|```', '', result).strip()
    return cleaned_result.lower()

def check_result_validity(result):
    try:
        is_literal = re.sub(r'\.', '', result).strip().lower() in ['yes', 'no']
        output = None
        output = Output.model_validate_json(clean_result(result))
        return is_literal or (output is not None)
    except Exception:
        return False

def get_y_hat(result) -> int:
    try:
        is_literal = re.sub(r'\.', '', result).strip().lower() in ['yes', 'no']
        if is_literal:
            return 1 if re.sub(r'\.', '', result).strip().lower() == 'yes' else 0

        output = Output.model_validate_json(clean_result(result))
        return 1 if output.contains_disinformation else 0

    except Exception:
        return -1

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

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\/:*?"<>|]', '_', filename)

def rq1(database: str):
    for model in model_inputs:
        print(f"processing model {model.model_name}", flush=True)
        model_result = process_model(model, database)
        sanitized_model = sanitize_filename(model.model_name)
        with open(f"rq1_{sanitized_model}.json", "w") as f:
            f.write(model_result.model_dump_json(indent=2))

        for prompt_type, results in model_result.row_results.items():
            print(f"{prompt_type}: Number of invalid results {len([i for i in results if i.invalid])}", flush=True)
            ys = [i.y for i in results if not i.invalid]
            y_hats = [i.y_hat for i in results if not i.invalid]

            tn, fp, fn, tp = confusion_matrix(ys, y_hats, labels=[0, 1]).ravel()

            print(f"{prompt_type}: True Negatives: {tn}", flush=True)
            print(f"{prompt_type}: False Positives: {fp}", flush=True)
            print(f"{prompt_type}: False Negatives: {fn}", flush=True)
            print(f"{prompt_type}: True Positives: {tp}", flush=True)

            # Calculate metrics
            accuracy = accuracy_score(ys, y_hats)
            precision = precision_score(ys, y_hats)  # Default: binary classification
            recall = recall_score(ys, y_hats)
            f1 = f1_score(ys, y_hats)

            # Print the results
            print(f"{prompt_type}: Accuracy: {accuracy:.2f}", flush=True)
            print(f"{prompt_type}: Precision: {precision:.2f}", flush=True)
            print(f"{prompt_type}: Recall: {recall:.2f}", flush=True)
            print(f"{prompt_type}: F1-score: {f1:.2f}", flush=True)

            # More detailed report
            print("{prompt_type}: Classification Report:\n", classification_report(ys, y_hats, labels=[0, 1], zero_division="warn"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq1.py <combined-db.sqlite>")
    rq1(sys.argv[1])