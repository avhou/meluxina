import sys
from pathlib import Path
import importlib
from models import *
import os

module_name = 'rq1-results-deepseek'
process_results_module = importlib.import_module(module_name)
process_results_deepseek = getattr(process_results_module, 'process_results')

module_name = 'rq1-results-deepseek-qwen'
process_results_module = importlib.import_module(module_name)
process_results_deepseek_qwen = getattr(process_results_module, 'process_results')

module_name = 'rq1-results-llama'
process_results_module = importlib.import_module(module_name)
process_results_llama = getattr(process_results_module, 'process_results')


known_models = [
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
]

model_process_results = [
   process_results_deepseek,
    process_results_deepseek_qwen,
    process_results_llama,
]

def process_results(input_folder: str):
    print(f"processing input folder {input_folder}")
    model_stats = []
    for input_file in Path(input_folder).rglob("*.json"):
        print(f"processing input file {input_file}")
        for (model, do_process_result) in zip(known_models, model_process_results):
            if sanitize_filename(model) in input_file.name:
                (file, stats) = do_process_result(input_file)
                model_stats.extend(stats)

    with open(os.path.join(input_folder, "rq1-table.md"), "w") as f:
        f.write(f"| Model | Accuracy | Precision | Recall | F1-score |\n")
        f.write(f"|:--|--:|--:|--:|--:|\n")
        for stat in model_stats:
            f.write(f"|{stat.model_name}|{stat.accuracy}|{stat.precision}|{stat.recall}|{stat.f1}|\n")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq1-results-deepseek.py <results_folder>")
    process_results(sys.argv[1])
