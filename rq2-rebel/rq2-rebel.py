import argparse
import sqlite3
from typing import List, Tuple, Union, Dict
from itertools import islice

import faiss
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from models import Output, Sample, Triple, TripleMetRowId, Metadata, MetadataList, SearchResult, PromptTemplate, PromptTemplates, Groupings

SAMPLE_FILE = "sample-rebel.json"
METADATA_FILE = "metadata-rebel.json"
PROMPTS_FILE = "prompts-rebel.json"
INDEX_FILE = "index-rebel.json"


def get_valid_triples(text: str) -> List[Triple]:
    text = text.replace("```json", "").replace("```", "")
    output = Output.model_validate_json(text)
    return [triple.normalize() for triple in output.triples if triple.subject is not None and triple.object is not None and triple.predicate is not None]


def generate_sample(ground_truth_db: str, chunk_db: str, max_chunks: int | None = None):
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
def get_embeddings_for_triples_met_rowid(triples: List[TripleMetRowId], tokenizer, model):
    return get_embeddings_for_triples([t.triple for t in triples], tokenizer, model)


# Function to extract vector embeddings for triples
def get_embeddings_for_triples(triples: List[Triple], tokenizer, model):
    embeddings = []

    for i, triple in enumerate(triples):
        # Create textual representation of the triple
        text = f"{triple.subject} {triple.predicate} {triple.object}"
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


def stats_sample(ground_truth_db: str):
    sample = read_sample()
    with sqlite3.connect(ground_truth_db) as conn:
        print(f"training")
        for disinformation, count in conn.execute(
            "SELECT disinformation, count(*) FROM articles WHERE rowid IN ({}) GROUP BY disinformation".format(",".join(map(str, sample.training)))
        ):
            print(f"disinformation {disinformation} count {count}")
        print(f"test")
        for disinformation, count in conn.execute(
            "SELECT disinformation, count(*) FROM articles WHERE rowid IN ({}) GROUP BY disinformation".format(",".join(map(str, sample.test)))
        ):
            print(f"disinformation {disinformation} count {count}")


def generate_index(ground_truth_db: str, chunk_db: str, max_chunks: int | None = None):
    print(f"processing chunk db {chunk_db}")

    sample = read_sample()
    print(str(sample))

    all_triples: List[TripleMetRowId] = []
    metadata: List[Metadata] = []
    with sqlite3.connect(chunk_db) as chunk_db_conn, sqlite3.connect(ground_truth_db) as ground_truth_db_conn:
        for row_id in islice(sample.training, 50_000_000 if max_chunks is None else max_chunks):
            ground_truth_url, ground_truth_disinformation, ground_truth_translated_text = ground_truth_db_conn.execute(
                "select url, disinformation, translated_text from articles where rowid = ?", (row_id,)
            ).fetchone()

            print(f"processing url {ground_truth_url} with disinformation {ground_truth_disinformation}")

            for url, chunk_number, chunk_triples, chunk_text, row_id in chunk_db_conn.execute(
                "select url, chunk_number, chunk_triples, chunk_text, rowid from chunked_articles where url = ? order by url, chunk_number", (ground_truth_url,)
            ):
                print(f"processing chunk {chunk_number} for url {url}, row_id {row_id}")

                try:
                    valid_triples = get_valid_triples(chunk_triples)
                    for i, triple in enumerate(valid_triples):
                        all_triples.append(TripleMetRowId(triple=triple, row_id=row_id))
                        metadata.append(
                            Metadata(
                                ground_truth_url=ground_truth_url,
                                ground_truth_disinformation=ground_truth_disinformation,
                                ground_truth_translated_text=ground_truth_translated_text,
                                chunked_db_row_id=row_id,
                                chunk_text=chunk_text,
                                triple=i + 1,
                                triple_text=f"{triple.subject} {triple.predicate} {triple.object}",
                            )
                        )
                except Exception as e:
                    print(f"error when parsing triples for url {url}, chunk_number {chunk_number} : {e}")

    print(f"found {len(all_triples)} valid triples")
    metadata_list = MetadataList(metadata=metadata)

    # metadata file is dus een mapping van de embedding index naar de metadata.
    # deze metadata bevat o.a. row id in de chunked articles database
    # als we dus een query doen op dichtbijzijnde vectoren, dan kunnen we deze mapping gebruiken om, gegeven de index
    # van de embedding, de row_id in de chunked articles database te krijgen, en aan de hand daarvan de text van die chunk te gaan ophalen
    with open(METADATA_FILE, "w") as f:
        f.write(metadata_list.model_dump_json())

    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

    embeddings = get_embeddings_for_triples_met_rowid(all_triples, tokenizer, model)
    embeddings = np.array(embeddings)

    print(f"embedding shape is {embeddings.shape}")
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(1024)
    index.add(normalized_embeddings)
    faiss.write_index(index, "index-rebel.bin")


def get_metadata() -> MetadataList:
    with open(METADATA_FILE, "r") as f:
        metadata = MetadataList.model_validate_json(f.read())
        return metadata


def get_top_search_results(
    results: List[SearchResult], metadata: List[Metadata], group_by: Groupings, take: Dict[Groupings, int]
) -> Tuple[List[float], List[Metadata]]:
    top_n = take[group_by]
    print(f"\nlooking for top {top_n} results grouped by {group_by}\n")
    score_position_pairs = [
        (score, position)
        for result in results
        for position, score in zip(result.training_nearest_embedding_positions, result.training_nearest_embedding_scores)
    ]
    score_position_pairs = sorted(score_position_pairs, key=lambda x: x[0], reverse=True)

    # Use a set to track distinct grouping cirteria.  we kunnen op eender wat groeperen.  op article url, op chunk url, op triple
    distinct_grouping_criterium = set()
    top_scores = []
    top_positions = []

    def get_grouping_criterium(metadata: Metadata):
        if group_by == "article":
            return metadata.ground_truth_url
        if group_by == "chunk":
            return metadata.chunked_db_row_id
        return f"{str(metadata.chunked_db_row_id).zfill(5)}{str(metadata.triple).zfill(5)}"

    for score, position in score_position_pairs:
        print(f"position {position} - score {score}")
        metadata_for_position = metadata[position]
        grouping_criterium = get_grouping_criterium(metadata_for_position)
        if grouping_criterium not in distinct_grouping_criterium:
            print(f"adding distinct criterium {grouping_criterium}")
            distinct_grouping_criterium.add(grouping_criterium)
            top_scores.append(score)
            top_positions.append(position)
        if len(top_positions) == top_n:
            break

    return (top_scores, [metadata[position] for position in top_positions])


def generate_prompts(ground_truth_db: str, chunk_db: str, top_k: int = 5, take: Dict[Groupings, int] = {"article": 5, "chunk": 5, "triple": 25}):
    print(f"prompts generation top-k {top_k}, top mappings {take}")
    sample = read_sample()
    print(str(sample))

    index = faiss.read_index("index-rebel.bin")
    print("index was read from disk")

    metadata = get_metadata()
    print("metadata was read from disk")

    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    print("model was instantiated")

    prompt_templates_article: List[PromptTemplate] = []
    prompt_templates_chunk: List[PromptTemplate] = []
    prompt_templates_triple: List[PromptTemplate] = []
    with sqlite3.connect(chunk_db) as chunk_db_conn, sqlite3.connect(ground_truth_db) as ground_truth_db_conn:
        for row_id in sample.test:
            search_results: List[SearchResult] = []

            # tuple syntax!
            (test_url, test_ground_truth, test_translated_text) = ground_truth_db_conn.execute(
                "select url, disinformation, translated_text from articles where rowid = ?", (row_id,)
            ).fetchone()

            print(f"processing test url {test_url}")

            triples_of_article: List[Triple] = []
            for url, chunk_number, chunk_triples, row_id in chunk_db_conn.execute(
                "select url, chunk_number, chunk_triples, rowid from chunked_articles where url = ? order by url, chunk_number", (test_url,)
            ):
                print(f"processing chunk {chunk_number} for url {url}, row_id {row_id}")

                try:
                    triples_of_article.extend(get_valid_triples(chunk_triples))

                except Exception as e:
                    print(f"error for url {url}, chunk_number {chunk_number} : {e}")

            query_vector = np.array(get_embeddings_for_triples(triples_of_article, tokenizer, model))
            print(f"shape of query vector is {query_vector.shape}")
            query_vector = query_vector / np.linalg.norm(query_vector)

            distances = []
            indices = []

            if len(triples_of_article) > 0:
                # hier krijgen we distances en indices terug voor alle triples
                distances, indices = index.search(query_vector, k=top_k)
                print(f"shape of distances is {distances.shape}, shape of indices is {indices.shape}")

            for idxs, scores in zip(indices, distances):
                search_results.append(
                    SearchResult(url=test_url, training_nearest_embedding_positions=list(idxs), training_nearest_embedding_scores=list(scores))
                )

            for group_by, prompt_templates in zip(["article", "chunk", "triple"], [prompt_templates_article, prompt_templates_chunk, prompt_templates_triple]):
                top_scores, top_metadata = get_top_search_results(search_results, metadata.metadata, group_by, take)
                print(f"top search scores grouped by {group_by} are {top_scores}")

                prompt_templates.append(
                    PromptTemplate(
                        url=test_url, article_text=test_translated_text, ground_truth_disinformation=test_ground_truth, metadata=top_metadata, scores=top_scores
                    )
                )

        with open(PROMPTS_FILE, "w") as f:
            f.write(
                PromptTemplates(
                    templates_article=prompt_templates_article, templates_chunk=prompt_templates_chunk, templates_triple=prompt_templates_triple
                ).model_dump_json()
            )


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
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        required=False,
        help="The top K most similar vectors that will be searched",
    )
    parser.add_argument(
        "--top-articles",
        type=int,
        default=5,
        required=False,
        help="The top K articles that will be injected in the LLM",
    )
    parser.add_argument(
        "--top-chunks",
        type=int,
        default=5,
        required=False,
        help="The top K chunks that will be injected in the LLM",
    )
    parser.add_argument(
        "--top-triples",
        type=int,
        default=25,
        required=False,
        help="The top K triples that will be injected in the LLM",
    )
    parser.add_argument("--generate-index", action="store_true", help="Flag to generate the index")
    parser.add_argument("--stats-sample", action="store_true", help="Flag to generate stats for the sample")
    parser.add_argument("--generate-sample", action="store_true", help="Flag to generate the sample")
    parser.add_argument("--generate-prompts", action="store_true", help="Flag to generate the prompts")
    args = parser.parse_args()

    if args.generate_sample:
        print(f"generating sample for {args.input_db} and input count {args.input_count}")
        generate_sample(args.ground_truth_db, args.input_db, args.input_count)

    if args.stats_sample:
        print(f"generating stats for sample for {args.input_db}")
        stats_sample(args.ground_truth_db)

    if args.generate_index:
        print(f"generating index for {args.input_db} and input count {args.input_count}")
        generate_index(args.ground_truth_db, args.input_db, args.input_count)

    if args.generate_prompts:
        print(f"generating prompts for {args.input_db}")
        generate_prompts(args.ground_truth_db, args.input_db, args.top_k, {"article": args.top_articles, "chunk": args.top_chunks, "triple": args.top_triples})

    print("done")
