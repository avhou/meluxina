from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter
import torch
import sys
import sqlite3
from models import Output, Triple, SubjectMetadata, TripleMetRowId
from typing import List
from unidecode import unidecode
from itertools import islice
import numpy as np
import faiss
from collections import defaultdict
import json
import argparse

# Function to extract vector embeddings for triples
def get_embeddings_for_triples(triples: List[TripleMetRowId], tokenizer, model):
    embeddings = []

    for i, triple in enumerate(triples):
        # Create textual representation of the triple
        text = f"{triple.triple.subject} {triple.triple.predicate} {triple.triple.object}"
        print(f"embedding for text {i+1}/{len(triples)}: {text}")

        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get the encoder's hidden states from the model
        with torch.no_grad():  # Disable gradient computation
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True
            )

        # Access the encoder's hidden states
        encoder_hidden_states = outputs.encoder_hidden_states  # Tuple of all encoder layers

        # Get the last layer's hidden state
        last_hidden_state = encoder_hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        # Compute the mean pooling of the last hidden state
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        embeddings.append(embedding)

    return embeddings

def extract_rebel_store(chunk_db: str, max_chunks: int = None):
    print(f"processing chunk db {chunk_db}")

    all_triples: List[TripleMetRowId] = []
    limit = "" if max_chunks is None else f" limit {max_chunks}"
    with sqlite3.connect(chunk_db) as conn:
        for row in conn.execute(
                f"select url, chunk_number, chunk_triples, rowid from chunked_articles {limit}"
        ):
            url = row[0]
            chunk_number = row[1]
            chunk_triples = row[2].replace("```json", "").replace("```", "")
            row_id = row[3]
            try:
                output = Output.model_validate_json(chunk_triples)
                valid_triples: List[Triple] = [
                    triple
                    for triple in output.triples
                    if triple.subject is not None
                       and triple.object is not None
                       and triple.predicate is not None
                ]
                for triple in valid_triples:
                    triple = triple.normalize()
                    all_triples.append(TripleMetRowId(triple=triple, row_id=row_id))
            except Exception as e:
                print(
                    f"error when parsing triples for url {url}, chunk_number {chunk_number} : {e}"
                )

    print(f"found {len(all_triples)} valid triples")
    mapping_embedding_index_to_row_id: List[int] = [t.row_id for t in all_triples]
    with open("metadata-rebel.json", "w") as f:
        json.dump(mapping_embedding_index_to_row_id, f)

    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    gen_kwargs = {
        "max_length": 1024,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 1,
    }

    embeddings = get_embeddings_for_triples(all_triples, tokenizer, model)
    embeddings = np.array(embeddings)

    print(f"embedding shape is {embeddings.shape}")
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(1024)
    index.add(normalized_embeddings)
    faiss.write_index(index, "index-rebel.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2")

    parser.add_argument("--input_db", type=str, required=True, help="Path to the input database file")
    parser.add_argument("--input_count", type=int, required=False, help="The number of input chunks to process")
    parser.add_argument("--generate_index", action="store_true", help="Flag to generate the index")
    args = parser.parse_args()
    print(args)
    extract_rebel_store(args.input_db, args.input_count)

