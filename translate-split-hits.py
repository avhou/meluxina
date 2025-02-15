import sys
import csv

import os
from transformers import MarianMTModel, MarianTokenizer

def generate_translation(input: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str :
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def do_translations(device: str, model: str, input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise ValueError(f"Input file not found: {input_file}")

    print(f"instantiating model")
    tokenizer = MarianTokenizer.from_pretrained(model)
    model = MarianMTModel.from_pretrained(model)
    model.to(device)
    print(f"model instantiated")

    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        reader_in = csv.reader(f_in)
        writer_out = csv.writer(f_out)
        # url, content
        for i, line in enumerate(reader_in):
            if i >= 5:
                break
            translation = generate_translation(line[1], tokenizer, model)
            print(f"translated: {line[0]} on device {device} with model {model}")
            writer_out.writerow([line[0], line[1], translation])
        print(f"done on device {device} with model {model}")



if __name__ == "__main__":
    if (len(sys.argv) < 5):
        print("Usage: python translate-split-hits.py device model input_file output_file")

    device = sys.argv[1]
    model = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    print(f"starten met de uitvoering van model {model} op device {device} voor input file {input_file}")