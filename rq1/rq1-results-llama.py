import sys
import json
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from models import *
import re


def process_results(input_file: str, suffix: Optional[str] = None) -> (str, List[ModelStats]):
    print(f"processing input {input_file}")
    with open(input_file, "r") as f:
        content = f.read()
        model_result = ModelResult.model_validate_json(content)

        if model_result is None:
            raise ValueError(f"Could not parse input file {input_file}")

        model_stats = []
        for prompt_type, row_results in model_result.row_results.items():
            print(f"processing {len(row_results)} results for prompt type {prompt_type}")
            sanitized_name = sanitize_filename(model_result.model_input.model_name)

            ys = [i.y for i in row_results]
            # print(f"y values: {ys}")
            y_hats = [get_y_hat(prompt_type, index, i.result, i.url) for (index, i) in enumerate(row_results)]
            # print(f"y_hat values: {y_hats}")
            y_hats_invalid = [i for i in y_hats if i == -1]
            # print(f"found {len(y_hats_invalid)} invalid y_hats")

            # voorlopig enkel met de valids verder gaan
            valid_indices = [index for index in range(len(y_hats)) if y_hats[index] != -1]
            print(f"prompt_type {prompt_type} has {len(valid_indices)} valid results and {len(ys) - len(valid_indices)} invalid results")

            valid_model_result = ModelResult(
                model_input=model_result.model_input,
                row_results={prompt_type: [row_results[i] for i in valid_indices]},
            )
            with open(f"rq1_{sanitized_name}_{prompt_type}_valid.json", "w") as f:
                f.write(valid_model_result.model_dump_json(indent=2))

            ys = [ys[i] for i in valid_indices]
            y_hats = [y_hats[i] for i in valid_indices]

            tn, fp, fn, tp = confusion_matrix(ys, y_hats, labels=[0, 1]).ravel()

            print(f"{prompt_type}: number of valid results: {len(y_hats)}", flush=True)
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
            print(
                f"\n{model_result.model_input.model_name}-{prompt_type}\n",
                classification_report(ys, y_hats, labels=[0, 1], zero_division="warn"),
                flush=True,
            )

            model_stats.append(
                ModelStats(
                    model_name=f"{model_result.model_input.model_name} - {prompt_type}"
                    if suffix is None
                    else f"{model_result.model_input.model_name} - {suffix} - {prompt_type}",
                    accuracy=f"{accuracy:.2f}",
                    precision=f"{precision:.2f}",
                    recall=f"{recall:.2f}",
                    f1=f"{f1:.2f}",
                )
            )
        result_file_name = f"rq1_{sanitized_name}_results.md" if suffix is None else f"rq1_{sanitized_name}_{suffix}_results.md"
        return result_file_name, model_stats


def remove_markdown(text: str) -> str:
    return text.replace("```turtle", "").replace("```json", "").replace("```", "")


def get_y_hat(prompt_type: PromptType, index: int, result: str, url: str) -> int:
    try:
        last_line = result.splitlines()[-1]
        parsed_result = json.loads(last_line)
        parsed_content = json.loads(remove_markdown(parsed_result["content"]))
        output = Output.model_validate(parsed_content)
        if output is not None:
            return 1 if output.contains_disinformation else 0
        else:
            print(f"exception for prompt_type {prompt_type} and index {index}: {e}")
            return -1
    except Exception as e:
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
                output = Output.model_validate_json(match)
                if output is not None:
                    return 1 if output.contains_disinformation else 0
                else:
                    print(f"match found but not valid json for prompt_type {prompt_type} and index {index}")
                    return -1
            else:
                print(f"no match found for prompt_type {prompt_type} and index {index} and url {url}")
                return -1
        except Exception as e:
            print(f"exception for prompt_type {prompt_type} and index {index}: {e}")
            return -1


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq1-results-llama.py <deepseek-results.sqlite>")
    (file, model_stats) = process_results(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    with open(file, "w") as f:
        f.write(f"| Model | Accuracy | Precision | Recall | F1-score |\n")
        f.write(f"|:--|--:|--:|--:|--:|\n")
        for stat in model_stats:
            f.write(f"|{stat.model_name}|{stat.accuracy}|{stat.precision}|{stat.recall}|{stat.f1}|\n")
