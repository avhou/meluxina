import sys
import csv
import itertools
import re
import torch

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

def chunk_text(text, device: str, max_length=250):
    sentences = re.split(r'\.|\?|!|\s\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = re.sub(r'\s+', ' ', sentence.strip()).strip()

        # If sentence is too long, split it by words
        if len(sentence) > max_length:
            print(f"device {device}: warning, sentence is too long, will split by words", flush=True)
            words = sentence.split()
            for i in range(0, len(words), max_length):
                split_sentence = " ".join(words[i:i + max_length]) + "."
                chunks.append(split_sentence.strip())
        else:
            # Normal chunking logic
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def translate_text(device: str, text: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str:
    text_chunks = chunk_text(text, device)

    translated_chunks = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated_ids = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
        translated_chunks.append(translated_text)

    return " ".join(translated_chunks)


def translate_text_batch(device: str, text: str, tokenizer: MarianTokenizer, model: MarianMTModel,
                   batch_size: int = 8) -> str:
    text_chunks = chunk_text(text, device)

    translated_chunks = []

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i: i + batch_size]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated_ids = model.generate(
                **inputs,
                max_length=512,            # Ensure output is bounded
                num_beams=3,               # Balanced between diversity and accuracy
                no_repeat_ngram_size=3,    # Prevents repetitive phrases
                early_stopping=True       # Stops when best translation is found
            )

        translated_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        translated_chunks.extend(translated_texts)

    return " ".join(translated_chunks)


def do_translations(device: str, model_name: str, input_file: str, output_file: str, batch_size: int = 8):
    if not os.path.exists(input_file):
        raise ValueError(f"Input file not found: {input_file}")

    print(f"instantiating model", flush=True)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)
    print(f"model instantiated", flush=True)

    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        reader_in = csv.reader(f_in)
        writer_out = csv.writer(f_out)
        lines = list(reader_in)
        total_nr_lines = len(lines)

        i = 1
        for line in lines:
            url = line[0]
            content = line[1]
            translation = translate_text_batch(device, content, tokenizer, model, batch_size)
            print(f"device {device} translated {i}/{total_nr_lines} : {url}", flush=True)
            writer_out.writerow([url, content, translation])
            f_out.flush()
            i = i + 1
        print(f"done on device {device} with model {model_name}", flush=True)



if __name__ == "__main__":
    if (len(sys.argv) < 5):
        print("Usage: python translate-split-hits.py device model input_file output_file")

    device = sys.argv[1]
    model_name = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    print(f"starten met de uitvoering van model {model_name} op device {device} voor input file {input_file}", flush=True)
    do_translations(device, model_name, input_file, output_file, 32)