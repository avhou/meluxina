import sqlite3
import sys
from llama_index.core.node_parser import SentenceSplitter
from typing import Callable
from pydantic import BaseModel
import transformers
import torch
from datetime import datetime
from models import Output


class ArticleChunk(BaseModel):
    article_url: str
    chunk_number: int
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
        for row in (
            conn.execute("select url, translated_text from articles")
            if use_translated_text
            else conn.execute("select url, text from articles")
        ):
            url = row[0]
            text = row[1]
            text = text.replace("\\s+", "")
            for index, chunk in enumerate(chunker.split_text(text)):
                callback(
                    ArticleChunk(article_url=url, chunk_number=index, chunk_text=chunk)
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

    def process_chunk(chunk: ArticleChunk):
        print(
            f"processing chunk {chunk.chunk_number}, url {chunk.article_url}, should write to {output_database}"
        )

        messages = [
            {
                "role": "system",
                "content": f"""You are an expert AI system that specializes in named entity recognition and knowledge graph extraction. 
            Your task is to analyse any provided input text, identify and extract the relevant entities and their attributes and relationships, 
            and output a knowledge graph.  
            You will output triples (subject, predicate, object) in JSON format.   
            The output should conform to this JSON schema : {Output.model_json_schema()}.  
            Only output the JSON-formatted knowledge graph and do not include any explanations.  
            Be as succinct as possible and only include the most relevant information.
            However, make sure to list just one concept per subject, predicate or object.  If a subject, predicate or object contains multiple concepts, split them into separate triples.
            """,
            },
            {"role": "user", "content": chunk.chunk_text},
        ]

        outputs = pipeline(messages, max_new_tokens=1024)
        json = outputs[0]["generated_text"][-1]["content"]
        print(
            f"chunk {chunk.chunk_number}, url {chunk.article_url} generated output {json}"
        )

    process_chunks_of_articles(input_database, process_chunk)


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("usage : rq2.py <combined-db.sqlite> <output-db.sqlite>")
    rq2_rdf2vec_preparation(sys.argv[1], sys.argv[2])
