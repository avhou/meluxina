from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.samplers import PageRankSampler
import sys
import sqlite3
from models import Output, Triple
from typing import List
from unidecode import unidecode
from pyrdf2vec.graphs import KG, Vertex


def normalize(value: str) -> str:
    return unidecode(value.lower())


def train_model(kg: KG, entities: List[str]):
    # max hop depth 4, geen limiet op nr of walks
    walker = RandomWalker(4, None, PageRankSampler(), random_state=1234)
    embedder = Word2Vec(epochs=10, workers=1)
    transformer = RDF2VecTransformer(embedder, walkers=[walker], verbose=1)

    # embeddings is een lijst van vectors, de size van de gevraagde entiteiten.  dus elke entiteit heeft een embedding vector
    embeddings, _ = transformer.fit_transform(kg, entities)
    return embeddings


def add_triples_to_kg(kg: KG, triples: List[Triple]):
    for triple in triples:
        subj = Vertex(normalize(triple.subject))
        obj = Vertex(normalize(triple.object))
        pred = Vertex(
            normalize(triple.predicate), predicate=True, vprev=subj, vnext=obj
        )
        kg.add_walk(subj, pred, obj)


def extract_rdf_store(chunk_db: str):
    print(f"processing chunk db {chunk_db}")

    seen = set()
    all_triples = []
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
                    triple_tuple = (triple.subject, triple.predicate, triple.object)
                    if triple_tuple not in seen:
                        seen.add(triple_tuple)
                        unique_triples.append(triple)
                all_triples.extend(unique_triples)
            except Exception as e:
                print(
                    f"error when parsing triples for url {url}, chunk_number {chunk_number} : {e}"
                )

    print(f"found {len(all_triples)} valid triples")

    kg = KG()
    add_triples_to_kg(kg, all_triples)

    unique_subjects = list({normalize(t.subject) for t in all_triples})
    embeddings = train_model(kg, unique_subjects)
    # for entity, embedding in zip(unique_subjects, embeddings):
    #     print(f"entiteit {entity} heeft embedding {embedding[:4]}")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2-ref2vec.py <chunk-db.sqlite>")
    extract_rdf_store(sys.argv[1])
