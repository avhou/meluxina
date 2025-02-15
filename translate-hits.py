import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sqlite3
from transformers import MarianMTModel, MarianTokenizer

def generate_translation(input: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str :
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def do_translations(data_dir: str, input_file: str, output_file: str):
    if not os.path.exists(input_file):
        raise ValueError(f"Input file not found: {input_file}")

    print(f"instantiating models")
    tokenizer_nl = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
    model_nl = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
    model_nl.to("cuda")
    tokenizer_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    model_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    model_fr.to("cuda")
    print(f"models instantiated")

    with sqlite3.connect(input_file) as input_conn, sqlite3.connect(output_file) as output_conn:
        output_conn.execute("drop table if exists hits_translation;")
        output_conn.execute("create table if not exists hits_translation (url text, content text, content_en, languages text);")

        print(f"reading relevant data")
        result = input_conn.execute("select url, content, languages from hits;").fetchall()
        urls = [r[0] for r in result]
        documents = [r[1] for r in result]
        languages = [r[2] for r in result]
        print(f"data was read, start processing")

        for i, doc in enumerate(documents):
            translation = None
            if ("fra" in languages[i]):
                print(f"document {i} is in french")
                translation = generate_translation(doc, tokenizer_fr, model_fr)
            elif ("nld" in languages[i]):
                print(f"document {i} is in dutch")
                translation = generate_translation(doc, tokenizer_nl, model_nl)
            else:
                print(f"document {i} will not be translated")
                translation = doc
            output_conn.execute("insert into hits_translation (url, content, content_en, languages) values (?, ?, ?, ?);", (urls[i], doc, translation, languages[i]))


if __name__ == "__main__":
    data_dir = os.environ["PROJECT_DATA_DIR"]
    input_file = os.environ["INPUT_FILE"]
    output_file = os.environ["OUTPUT_FILE"]

    if (data_dir is None):
        raise ValueError("Data directory not set")
    if (input_file is None):
        raise ValueError("Input file not set")
    if (output_file is None):
        raise ValueError("Output file not set")

    do_translations(data_dir, os.path.join(data_dir, input_file), os.path.join(data_dir, output_file))