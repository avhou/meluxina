import sys
import csv
import itertools
import re
import torch

from typing import List
import os
from transformers import MarianMTModel, MarianTokenizer
from llama_index import SentenceSplitter

csv.field_size_limit(10 * 1024 * 1024)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)  # Remove weird symbols
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


def chunk_text(text, device: str, max_words=250):
    splitter = SentenceSplitter(chunk_size=max_words, chunk_overlap=50)
    # sentences = re.split(r'\.|\?|!', text)
    sentences = splitter.split(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # words = clean_text(sentence).strip().split()  # Split sentence into words
        words = sentence.strip().split()  # Split sentence into words
        num_words = len(words)

        if num_words > max_words:  # If sentence is too long, split further
            print(f"device {device}: warning, sentence is too long, will split further.  sentence is {sentence}", flush=True)

            # eerst flushen van de eventuele huidige chunk
            if len(current_chunk) > 0:
                chunks.append(" ".join(current_chunk).strip() + ".")
                current_chunk = []
                current_length = 0

            for i in range(0, num_words, max_words):
                split_sentence = " ".join(words[i:i + max_words]) + "."  # Add period
                chunks.append(split_sentence.strip())
        else:
            if current_length + num_words <= max_words:
                current_chunk.extend(words)
                current_length += num_words
            else:
                chunks.append(" ".join(current_chunk).strip() + ".")
                current_chunk = words
                current_length = num_words

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip() + ".")

    return chunks

def get_max_output_length(inputs, scale_factor=1.3, max_len=512):
    # Find the longest input sequence in the batch
    max_input_length = max(len(input_ids) for input_ids in inputs['input_ids'])

    # Calculate the max output length based on a scaling factor
    max_output_length = min(max_len, int(max_input_length * scale_factor))

    return max_output_length

def translate_text_batch(device: str, text: str, tokenizer: MarianTokenizer, model: MarianMTModel,
                   batch_size: int = 8) -> str:
    text_chunks = chunk_text(text, device)

    translated_chunks = []

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i: i + batch_size]

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        max_output_length = get_max_output_length(inputs)
        with torch.no_grad():
            translated_ids = model.generate(
                **inputs,
                max_length=max_output_length,
                do_sample=True,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
                repetition_penalty=2.0,
                length_penalty=1.1,
                temperature=0.1,
                top_k=25,
                pad_token_id=tokenizer.pad_token_id
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
            if i > 10:
                break
        print(f"done on device {device} with model {model_name}", flush=True)



if __name__ == "__main__":
    if (len(sys.argv) < 5):
        print("Usage: python translate-split-hits.py device model input_file output_file")

    device = sys.argv[1]
    model_name = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    print(f"starten met de uitvoering van model {model_name} op device {device} voor input file {input_file}", flush=True)
    do_translations(device, model_name, input_file, output_file, 4)