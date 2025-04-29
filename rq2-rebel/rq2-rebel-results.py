import argparse
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from models import ModelResult, ModelResults, ModelStats, ClassificationOutput
from pathlib import Path
import re
import json


def get_model_results(file: Path) -> List[ModelResult]:
    with open(file, "r") as f:
        return ModelResults.model_validate_json(f.read()).results


def get_file_info(file: str) -> Tuple[str, str]:
    pattern = r"rq2-results-group-by-(\w+)-(.+)\.json"
    match = re.match(pattern, file)
    if match:
        group_by = match.group(1)
        model = match.group(2)
        return (group_by, model)
    raise ValueError(f"could not parse info for {file}")


def get_y_hat(model: str, index: int, result: str) -> int:
    try:
        result = json.loads(result)["content"]
        result = result.replace("\n", "")
        result = result.replace("'", '"')
    except Exception as e:
        pass
    try:
        matches = re.findall(r"(?:```json)?\s*(\{.*?\})\s*(?:```)?", result, re.DOTALL)
        if matches:
            match = matches[-1].strip()
            match = match.replace("\n", "")
            match = match.replace("'", '"')
            output = ClassificationOutput.model_validate_json(match)
            if output is not None:
                return 1 if output.contains_disinformation else 0
            else:
                print(f"match found but not valid json for model {model} and index {index}")
                return -1
        else:
            print(f"no match found for model {model} and index {index}")
            return -1
    except Exception as e:
        print(f"exception for model {model} and index {index}: {e}")
        return -1


def generate_model_stats(model: str, results: List[ModelResult]) -> ModelStats:
    ys = [i.y for i in results]
    y_hats = [get_y_hat(model, index, i.result) for (index, i) in enumerate(results)]

    valid_indices = [index for index in range(len(y_hats)) if y_hats[index] != -1]
    print(f"model {model} has {len(valid_indices)} valid results")

    ys = [ys[i] for i in valid_indices]
    y_hats = [y_hats[i] for i in valid_indices]

    tn, fp, fn, tp = confusion_matrix(ys, y_hats, labels=[0, 1]).ravel()

    print(f"{model}: number of valid results: {len(y_hats)}", flush=True)
    # print(f"{model}: True Negatives: {tn}", flush=True)
    # print(f"{model}: False Positives: {fp}", flush=True)
    # print(f"{model}: False Negatives: {fn}", flush=True)
    # print(f"{model}: True Positives: {tp}", flush=True)

    # Calculate metrics
    accuracy = accuracy_score(ys, y_hats)
    precision = precision_score(ys, y_hats)  # Default: binary classification
    recall = recall_score(ys, y_hats)
    f1 = f1_score(ys, y_hats)

    # Print the results
    # print(f"{model}: Accuracy: {accuracy:.2f}", flush=True)
    # print(f"{model}: Precision: {precision:.2f}", flush=True)
    # print(f"{model}: Recall: {recall:.2f}", flush=True)
    # print(f"{model}: F1-score: {f1:.2f}", flush=True)

    # More detailed report
    # print(
    #     f"\n{model}\n",
    #     classification_report(ys, y_hats, labels=[0, 1], zero_division="warn"),
    #     flush=True,
    # )

    return ModelStats(
        model_name=model,
        accuracy=f"{accuracy:.2f}",
        precision=f"{precision:.2f}",
        recall=f"{recall:.2f}",
        f1=f"{f1:.2f}",
    )


def write_model_stats(model_stats: List[ModelStats]):
    with open("rq2-results-table.md", "w") as f:
        f.write(f"| Model | Accuracy | Precision | Recall | F1-score |\n")
        f.write(f"|:--|--:|--:|--:|--:|\n")
        for stat in model_stats:
            f.write(f"|{stat.model_name}|{stat.accuracy}|{stat.precision}|{stat.recall}|{stat.f1}|\n")


def generate_stats(input_folder: str):
    model_stats: List[ModelStats] = []
    for file in Path(input_folder).rglob("rq2-results*.json"):
        (group_by, model) = get_file_info(file.name)
        model_results = get_model_results(file)
        print(f"found {len(model_results)} results for model {model} with group by {group_by}")
        model_stats.append(generate_model_stats(f"{model}-group-by-{group_by}", model_results))
    write_model_stats(model_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 results")

    parser.add_argument("--results-folder", type=str, required=True, help="Path to the results folder")
    args = parser.parse_args()

    generate_stats(args.results_folder)

    print("done")
