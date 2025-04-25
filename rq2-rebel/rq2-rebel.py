import argparse
import json
import sqlite3
from typing import List
from itertools import islice

import faiss
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from models import Output, Sample, Triple, TripleMetRowId

SAMPLE_FILE = "sample-rebel.json"
METADATA_FILE = "metadata-rebel.json"
INDEX_FILE = "index-rebel.json"


def generate_sample(ground_truth_db: str, chunk_db: str, max_chunks: int = None):
    limit = "" if max_chunks is None else f" limit {max_chunks}"
    with sqlite3.connect(ground_truth_db) as conn:
        rows = [
            (row[0], row[1])  # row[0] = rowid, row[1] = disinformation label (1 or 0)
            for row in conn.execute(f"select rowid, case when disinformation = 'y' then 1 else 0 end as label from articles {limit}")
        ]

    # Extract row IDs and labels
    xs, ys = zip(*rows)

    # Perform a stratified split (75% training, 25% testing)
    train_xs, test_xs, train_ys, test_ys = train_test_split(xs, ys, test_size=0.25, stratify=ys, random_state=42)

    print(f"Training set size: {len(train_xs)}, Testing set size: {len(test_xs)}")
    print(f"Training label distribution: {dict(zip(*np.unique(train_ys, return_counts=True)))}")
    print(f"Testing label distribution: {dict(zip(*np.unique(test_ys, return_counts=True)))}")

    with open(SAMPLE_FILE, "w") as f:
        f.write(Sample(training=train_xs, test=test_xs).model_dump_json())


# Function to extract vector embeddings for triples
def get_embeddings_for_triples(triples: List[TripleMetRowId], tokenizer, model):
    embeddings = []

    for i, triple in enumerate(triples):
        # Create textual representation of the triple
        text = f"{triple.triple.subject} {triple.triple.predicate} {triple.triple.object}"
        print(f"embedding for text {i + 1}/{len(triples)}: {text}")

        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get the encoder's hidden states from the model
        with torch.no_grad():  # Disable gradient computation
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        # Access the encoder's hidden states
        encoder_hidden_states = outputs.encoder_hidden_states  # Tuple of all encoder layers

        # Get the last layer's hidden state
        last_hidden_state = encoder_hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        # Compute the mean pooling of the last hidden state
        embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        embeddings.append(embedding)

    return embeddings


def read_sample() -> Sample:
    with open(SAMPLE_FILE, "r") as f:
        sample = Sample.model_validate_json(f.read())
        return sample


def generate_index(ground_truth_db: str, chunk_db: str, max_chunks: int = None):
    print(f"processing chunk db {chunk_db}")

    sample = read_sample()
    print(str(sample))

    all_triples: List[TripleMetRowId] = []
    with sqlite3.connect(chunk_db) as chunk_db_conn, sqlite3.connect(ground_truth_db) as ground_truth_db_conn:
        for row_id in islice(sample.training, 50_000_000 if max_chunks is None else max_chunks):
            ground_truth_url = ground_truth_db_conn.execute(f"select url from articles where rowid = ?", (row_id,)).fetchone()[0]

            for row in chunk_db_conn.execute(
                f"select url, chunk_number, chunk_triples, rowid from chunked_articles where url = ? order by url, chunk_number", (ground_truth_url,)
            ):
                url = row[0]
                chunk_number = row[1]
                chunk_triples = row[2].replace("```json", "").replace("```", "")
                row_id = row[3]

                try:
                    output = Output.model_validate_json(chunk_triples)
                    valid_triples: List[Triple] = [
                        triple for triple in output.triples if triple.subject is not None and triple.object is not None and triple.predicate is not None
                    ]
                    for triple in valid_triples:
                        triple = triple.normalize()
                        all_triples.append(TripleMetRowId(triple=triple, row_id=row_id))
                except Exception as e:
                    print(f"error when parsing triples for url {url}, chunk_number {chunk_number} : {e}")

    print(f"found {len(all_triples)} valid triples")
    mapping_embedding_index_to_row_id: List[int] = [t.row_id for t in all_triples]
    with open(METADATA_FILE, "w") as f:
        json.dump(mapping_embedding_index_to_row_id, f)

    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    embeddings = get_embeddings_for_triples(all_triples, tokenizer, model)
    embeddings = np.array(embeddings)

    print(f"embedding shape is {embeddings.shape}")
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(1024)
    index.add(normalized_embeddings)
    faiss.write_index(index, "index-rebel.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2")

    parser.add_argument("--input-db", type=str, required=True, help="Path to the input database file")
    parser.add_argument(
        "--ground-truth-db",
        type=str,
        required=True,
        help="Path to the ground truth database file",
    )
    parser.add_argument(
        "--input-count",
        type=int,
        required=False,
        help="The number of input chunks to process",
    )
    parser.add_argument("--generate-index", action="store_true", help="Flag to generate the index")
    parser.add_argument("--generate-sample", action="store_true", help="Flag to generate the sample")
    args = parser.parse_args()

    if args.generate_sample:
        print(f"generating sample for {args.input_db} and input count {args.input_count}")
        generate_sample(args.ground_truth_db, args.input_db, args.input_count)

    if args.generate_index:
        print(f"generating index for {args.input_db} and input count {args.input_count}")
        generate_index(args.ground_truth_db, args.input_db, args.input_count)

    print("done")
