import sys
import csv

from typing import List
import os
from transformers import MarianMTModel, MarianTokenizer

csv.field_size_limit(10 * 1024 * 1024)

def generate_translation(input: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str :
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def generate_translations(input: List[str], tokenizer: MarianTokenizer, model: MarianMTModel) -> List[str] :
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    translated = model.generate(**inputs)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return result

def do_translations(device: str, model_name: str, input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise ValueError(f"Input file not found: {input_file}")

    print(f"instantiating model")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)
    print(f"model instantiated")

    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        reader_in = csv.reader(f_in)
        writer_out = csv.writer(f_out)
        # url, content
        lines = list(reader_in)
        chunk_size = 16
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            translations = generate_translations([line[1] for line in chunk], tokenizer, model)
            print(f"translated: {len(translations)} on device {device} with model {model_name}")
            for line, translation in zip(chunk, translations):
                writer_out.writerow([line[0], line[1], translation])
        print(f"done on device {device} with model {model_name}")



if __name__ == "__main__":
    if (len(sys.argv) < 5):
        print("Usage: python translate-split-hits.py device model input_file output_file")

    device = sys.argv[1]
    model_name = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    print(f"starten met de uitvoering van model {model_name} op device {device} voor input file {input_file}")
    do_translations(device, model_name, input_file, output_file)