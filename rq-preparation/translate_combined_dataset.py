import asyncio
import sqlite3
import re
from itertools import islice
from googletrans import Translator
from llama_index.core.node_parser import SentenceSplitter
import sys

# Configurable limits
MAX_WORDS = 500  # Chunk size in words
OVERLAP = 0  # Overlapping words to maintain context
BATCH_SIZE = 10  # Rows to process in one batch
MAX_CONCURRENT_REQUESTS = 3  # Limit simultaneous translations

# Initialize the LlamaIndex Sentence Splitter
splitter = SentenceSplitter(chunk_size=MAX_WORDS, chunk_overlap=OVERLAP)

async def translate_text(translator, text, lang, semaphore):
    """Translates text using LlamaIndex chunking for large inputs."""
    async with semaphore:
        try:
            # Split text into chunks
            chunks = splitter.split_text(text)

            # Translate each chunk separately
            translated_chunks = []
            for chunk in chunks:
                print(f"translating chunk: {chunk[:100]}")
                translated = await translator.translate(chunk, src=lang, dest='en')
                translated_chunks.append(translated.text)

            return " ".join(translated_chunks)  # Rejoin translated parts
        except Exception as e:
            return f"ERROR: {e}"

async def translate_combined_datasets_async(dataset: str):
    translator = Translator()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # Limit concurrent requests

    with sqlite3.connect(dataset) as conn:
        cursor = conn.execute("SELECT url, text, detected_language FROM articles ORDER BY url ASC")

        while True:
            batch = list(islice(cursor, BATCH_SIZE))  # Fetch rows in chunks
            if not batch:
                break  # Stop when no more rows

            tasks = [translate_text(translator, re.sub(r'\s+', ' ', text), lang, semaphore)
                     for _, text, lang in batch]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (url, _, _), translated in zip(batch, results):
                if "ERROR:" in translated:
                    print(f"Could not translate {url}, exception: {translated}")
                else:
                    print(f"Translated [{url}]: {translated[:100]}")  # Preview first 100 chars
                    conn.execute(f"update articles set translated_text = ? where url = ?", (translated, url))
                    conn.commit()

def translate_combined_datasets(dataset: str):
    """Run async translation in a sync environment"""
    asyncio.run(translate_combined_datasets_async(dataset))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : translate_combined_dataset.py <combined-db.sqlite>")
    translate_combined_datasets(sys.argv[1])
