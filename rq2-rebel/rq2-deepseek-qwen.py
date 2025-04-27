import os


from rq2_generic import get_arg_parser, process_prompts, create_model
from models import ClassificationOutput

print(f"found HUGGINGFACE_HUB_CACHE : {os.environ.get('HUGGINGFACE_HUB_CACHE')}", flush=True)
print(f"found HF_HOME : {os.environ.get('HF_HOME')}", flush=True)
print(f"found HUGGINGFACEHUB_API_TOKEN : {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}", flush=True)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
MODEL_PARAMS = {}


def model_generator():
    return create_model(MODEL_NAME, MODEL_PARAMS)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    process_prompts(
        args.prompts,
        args.group_by,
        model_generator,
        "deepseek-qwen",
        f"""
Follow these instructions carefully.
A user will submit articles related to immigration (in the broad sense) to you, and you have to determine whether the article contains disinformation
(that is, intentionally fabricated or false information shared with the aim of deceiving, manipulating opinions, promoting specific agendas, or reinforcing prejudices).
Use the provided context to guide your classification.
Answer with a simple true or false, true if you think the article contains disinformation, false if you think the article does not contain disinformation.
Do not give any further explanation or justification. 
Keep your reasoning output to a minimum.  
Be as succinct as possible.  
The only relevant output is your final classification.  
Generate your output in JSON format.  
The output should conform to this JSON schema : {ClassificationOutput.model_json_schema()}.
""",
    )

    print("done", flush=True)
