from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.samplers import PageRankSampler
import sys
import sqlite3
from models import Output, Triple, SubjectMetadata
from typing import List
from unidecode import unidecode
from pyrdf2vec.graphs import KG, Vertex
from itertools import islice
import numpy as np
import faiss
from collections import defaultdict



def normalize(value: str) -> str:
    return unidecode(value.lower())


def train_model(kg: KG, entities: List[str]):
    # max hop depth 4, geen limiet op nr of walks
    walker = RandomWalker(4, None, PageRankSampler(), random_state=1234)
    embedder = Word2Vec(epochs=10, workers=1)
    transformer = RDF2VecTransformer(embedder, walkers=[walker], verbose=1)

    # embeddings is een lijst van vectors, de size van de gevraagde entiteiten.  dus elke entiteit heeft een embedding vector
    embeddings, _ = transformer.fit_transform(kg, entities)
    return embeddings, embedder


def add_triples_to_kg(kg: KG, triples: List[Triple]):
    for triple in triples:
        subj = Vertex(normalize(triple.subject))
        obj = Vertex(normalize(triple.object))
        pred = Vertex(
            normalize(triple.predicate), predicate=True, vprev=subj, vnext=obj
        )
        kg.add_walk(subj, pred, obj)


def group_metadata_by_index(metadata_list: List[SubjectMetadata]) -> List[List[SubjectMetadata]]:
    grouped_metadata = defaultdict(list)

    # Group metadata by index
    for metadata in metadata_list:
        grouped_metadata[metadata.index].append(metadata)

    # Convert to a list of lists, ensuring the order of indices
    max_index = max(grouped_metadata.keys())
    result = [grouped_metadata[i] for i in range(max_index + 1)]

    return result

def extract_rdf_store(chunk_db: str):
    print(f"processing chunk db {chunk_db}")

    seen_triples = set()
    seen_entities = set()
    all_triples = []
    unique_subjects = []
    subject_to_metadata_mapping = {}
    with sqlite3.connect(chunk_db) as conn:
        for row in conn.execute(
            "select url, chunk_number, chunk_triples from chunked_articles"
        ):
            url = row[0]
            chunk_number = row[1]
            chunk_triples = row[2].replace("```json", "").replace("```", "")
            try:
                output = Output.model_validate_json(chunk_triples)
                valid_triples = [
                    triple
                    for triple in output.triples
                    if triple.subject is not None
                    and triple.object is not None
                    and triple.predicate is not None
                ]
                unique_triples = []
                for triple in valid_triples:
                    s, p, o = (
                        normalize(triple.subject),
                        normalize(triple.predicate),
                        normalize(triple.object),
                    )
                    if (s, p, o) not in seen_triples:
                        seen_triples.add((s, p, o))
                        unique_triples.append(triple)

                    if s not in seen_entities:
                        seen_entities.add(s)
                        unique_subjects.append(s)

                    position = unique_subjects.index(s)
                    metadata = SubjectMetadata(
                        subject=s,
                        index=position,
                        url=url,
                        chunk_number=chunk_number,
                        triple=(s, p, o),
                    )
                    if s not in subject_to_metadata_mapping.keys():
                        subject_to_metadata_mapping[s] = [metadata]
                    else:
                        subject_to_metadata_mapping[s].append(metadata)

                all_triples.extend(unique_triples)
            except Exception as e:
                print(
                    f"error when parsing triples for url {url}, chunk_number {chunk_number} : {e}"
                )

    print(f"found {len(all_triples)} valid triples")
    metadata_flat_list = group_metadata_by_index([metadata for metadata_list in subject_to_metadata_mapping.values() for metadata in metadata_list])
    print(f"first metadata : {metadata_flat_list[0]}")

    kg = KG()
    add_triples_to_kg(kg, all_triples)

    embeddings, embedder = train_model(kg, unique_subjects)
    for entity, embedding in islice(zip(unique_subjects, embeddings), 10):
        print(f"entiteit {entity} heeft embedding {embedding[:4]}")
    embeddings = np.array(embeddings)

    print(f"vector size is {embedder._model.vector_size}")
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embedder._model.vector_size)
    index.add(normalized_embeddings)

    query_vector = np.array([embeddings[0]])  # shape: (1, vector_size)
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search
    D, I = index.search(query_vector, k=3)  # D: distances, I: indices

    # Results
    for idx, score in zip(I[0], D[0]):
        print(f"Entity ID: {idx}")
        print(f"Metadata: {metadata_flat_list[idx]}")
        print(f"Score: {score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2-ref2vec.py <chunk-db.sqlite>")
    extract_rdf_store(sys.argv[1])
