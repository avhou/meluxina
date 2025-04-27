import sqlite3
import sys
from llama_index.core.node_parser import SentenceSplitter
from typing import Callable
from pydantic import BaseModel
import transformers
import torch
from datetime import datetime
from models import Output
import re


class ArticleChunk(BaseModel):
    article_url: str
    article_number: int
    article_number_total: int
    chunk_number: int
    chunk_number_total: int
    chunk_text: str


def process_chunks_of_articles(
    input_database: str,
    callback: Callable[[ArticleChunk], None],
    max_words: int = 250,
    overlap: int = 0,
    use_translated_text: bool = True,
):
    print(f"processing input database {input_database}")
    chunker = SentenceSplitter(chunk_size=max_words, chunk_overlap=overlap)
    with sqlite3.connect(input_database) as conn:
        article_number_total = conn.execute(
            "select count(url) from articles_reddit"
        ).fetchone()[0]
        article_number = 0
        for row in (
            conn.execute("select url, translated_text from articles_reddit")
            if use_translated_text
            else conn.execute("select url, text from articles_reddit")
        ):
            url = row[0]
            text = row[1]
            text = text.replace("\\s+", "")
            article_number = article_number + 1

            chunks = chunker.split_text(text)

            for index, chunk in enumerate(chunks):
                callback(
                    ArticleChunk(
                        article_url=url,
                        article_number=article_number,
                        article_number_total=article_number_total,
                        chunk_number=index + 1,
                        chunk_number_total=len(chunks),
                        chunk_text=chunk,
                    )
                )


def rq2_rdf2vec_preparation(input_database: str, output_database: str):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using device {device}", flush=True)

    print(
        f"""Starting model load at {datetime.now().strftime("%H:%M:%S")}""", flush=True
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/phi-4",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    print(
        f"""Done loading model at {datetime.now().strftime("%H:%M:%S")}""", flush=True
    )

    def process_chunk(chunk: ArticleChunk, conn):
        print(
            f"{datetime.now().strftime('%H:%M:%S')}: start processing chunk {chunk.chunk_number}/{chunk.chunk_number_total}, article {chunk.article_number}/{chunk.article_number_total}, url {chunk.article_url}"
        )

        bestaat = conn.execute(
            "select exists(select 1 from chunked_articles where url=? and chunk_number=?)",
            (chunk.article_url, chunk.chunk_number),
        ).fetchone()[0]

        if bestaat == 1:
            print(
                f"{datetime.now().strftime('%H:%M:%S')}: skip chunk {chunk.chunk_number}/{chunk.chunk_number_total}, article {chunk.article_number}/{chunk.article_number_total}, url {chunk.article_url}, already done"
            )
        else:
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
                Your task is to analyse any provided input text, identify and extract the relevant entities and their attributes and relationships, 
                and output a knowledge graph.  
                You will output triples (subject, predicate, object) in JSON format.   
                The output should conform to this JSON schema : {Output.model_json_schema()}.  
                Only output the JSON-formatted knowledge graph and do not include any explanations.  
                Be as succinct as possible and only include the most relevant information, while maintaining coherence in the knowledge graph.
                Make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
                If you are unable to extract any triples, e.g. due to very short input texts, just return the output in the correct format with an empty triples array.
                """,
                },
                {"role": "user", "content": chunk.chunk_text},
            ]

            outputs = pipeline(messages, max_new_tokens=1024)
            json = outputs[0]["generated_text"][-1]["content"]
            json = f"{json}".replace("\n", " ").replace("\r", " ")
            json = re.sub("\\s+", " ", json)
            conn.execute(
                "insert into chunked_articles(url, chunk_number, chunk_text, chunk_triples) values (?, ?, ?, ?)",
                (chunk.article_url, chunk.chunk_number, chunk.chunk_text, json),
            )
            conn.commit()
            print(
                f"{datetime.now().strftime('%H:%M:%S')}: {chunk.chunk_number}/{chunk.chunk_number_total}, article {chunk.article_number}/{chunk.article_number_total}, url {chunk.article_url} generated output {json}"
            )

    with sqlite3.connect(output_database) as conn:
        conn.execute(
            "create table if not exists chunked_articles(url text, chunk_number int, chunk_text text, chunk_triples text)"
        )
        process_chunks_of_articles(
            input_database,
            lambda chunk: process_chunk(chunk, conn),
            max_words=250,
            overlap=0,
        )


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("usage : rq2.py <combined-db.sqlite> <output-db.sqlite>")
    rq2_rdf2vec_preparation(sys.argv[1], sys.argv[2])
