from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter
import torch
import sys
import sqlite3
from models import Output, Triple, SubjectMetadata
from typing import List
from unidecode import unidecode
from itertools import islice
import numpy as np
import faiss
from collections import defaultdict
import json

# def extract_triplets(text):
#     print(f"received text : {text}")
#     triplets = []
#     relation, subject, relation, object_ = '', '', '', ''
#     text = text.strip()
#     current = 'x'
#     for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
#         if token == "<triplet>":
#             current = 't'
#             if relation != '':
#                 triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
#                 relation = ''
#             subject = ''
#         elif token == "<subj>":
#             current = 's'
#             if relation != '':
#                 triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
#             object_ = ''
#         elif token == "<obj>":
#             current = 'o'
#             relation = ''
#         else:
#             if current == 't':
#                 subject += ' ' + token
#             elif current == 's':
#                 object_ += ' ' + token
#             elif current == 'o':
#                 relation += ' ' + token
#     if subject != '' and relation != '' and object_ != '':
#         triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
#     return triplets
#
# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
# model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
# gen_kwargs = {
#     "max_length": 1024,
#     "length_penalty": 0,
#     "num_beams": 3,
#     "num_return_sequences": 1,
# }
#
# # Text to extract triplets from
# text = """
# On July 27, the Commission informed the city council about the reception of refugees who will be received in temporary emergency housing at the Prof. dr. E.M. Meijerslaan.
# On July 20, 2022, State Secretary Eric van der Burg indicated that the cabinet wants to allow municipalities that they do have to compensate for Ukrainians, but not asylum seekers from other countries.
# Wil Roode, Citizen Council member SP: "When on July 29, 2022, the Human Rights League brought out a statement that the cabinet thus legitimizes an unequal treatment of refugees and asylum seekers, I immediately thought that the College van Amstelveen has already done this and very quickly, namely very shortly after the ruling of 20 July. Because on July 25, refugees from the Ukraine have already been placed in the temporary emergency shelter."
# Such a preferred policy is discriminatory and undesirable.
# In the Netherlands there are four thousand municipal beds for the reception of Ukrainians unoccupied, while in Ter Apel there are regularly asylum seekers in the grass.
# With policy only to offer care to people from the Ukraine, a municipality immediately distinguishes based on origin between people who flee Ukraine and asylum seekers from other countries.
# Wil Roode: "Not only there are already 26 selected families from the Ukraine in the emergency shelter, but the college has used selection criteria without submitting it to the city council.
# The SP wants to know why selections have taken place and whether these criteria have been tested. On our own selecting we find incorrect."
# The SP therefore asks written questions for clarification about this issue because the Commission runs the risk because of selecting on the basis of origin to be guilty of unequal treatment.
# """
#
# # text = 'Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic.'
#
#
# MAX_WORDS = 200  # Chunk size in words
# OVERLAP = 10  # Overlapping words to maintain context
#
# # Initialize the LlamaIndex Sentence Splitter
# splitter = SentenceSplitter(chunk_size=MAX_WORDS, chunk_overlap=OVERLAP)
# chunks = splitter.split_text(text)

# Process each chunk
# all_triplets = []
# for chunk in chunks:
#     print(f"Processing chunk: {chunk}")
#     model_inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True)
#     generated_tokens = model.generate(
#         model_inputs["input_ids"].to(model.device),
#         attention_mask=model_inputs["attention_mask"].to(model.device),
#         **gen_kwargs,
#     )
#     decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
#     for sentence in decoded_preds:
#         print(f"Decoded sentence: {sentence}")
#         all_triplets.extend(extract_triplets(sentence))
#
# # Extract triplets
# for idx, sentence in enumerate(all_triplets):
#     print(f'Prediction triplets sentence {idx}')
#     print(sentence)
#
#
# Function to extract vector embeddings for triples
def get_embeddings_for_triples(triples: List[Triple], tokenizer, model):
    embeddings = []

    for triple in triples:
        # Create textual representation of the triple
        text = f"{triple.subject} {triple.predicate} {triple.object}"
        print(f"embedding for text: {text}")

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

def get_embeddings_for_subject(subjects: List[str], tokenizer, model):
    embeddings = []

    for i, subject in enumerate(subjects):
        # Create textual representation of the triple
        print(f"embedding for subject ({i + 1}/{len(subjects)}): {subject}")

        # Tokenize the text
        inputs = tokenizer(subject, return_tensors="pt", padding=True, truncation=True)

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

# # Example triples
# triples = [
#     {"subject": "Wil Roode", "predicate": "stated", "object": "refugees"},
#     {"subject": "Wil Roode", "predicate": "stated", "object": "housing"},
#     {"subject": "SP", "predicate": "wants", "object": "clarification"}
# ]
#
# # model.eval()
# # Get embeddings for the triples
# embeddings = get_embeddings_for_triples(triples)
#
# # Print the embeddings
# for idx, embedding in enumerate(embeddings):
#     print(f"Embedding for triple {idx+1} with dimension {embedding.shape}:")
#     print(embedding)
#



def group_metadata_by_index(metadata_list: List[SubjectMetadata]) -> List[List[SubjectMetadata]]:
    grouped_metadata = defaultdict(list)

    # Group metadata by index
    for metadata in metadata_list:
        grouped_metadata[metadata.index].append(metadata)

    # Convert to a list of lists, ensuring the order of indices
    max_index = max(grouped_metadata.keys())
    result = [grouped_metadata[i] for i in range(max_index + 1)]

    return result


def write_metadata_to_json(metadata: List[List[SubjectMetadata]], output_file: str):
    # Convert the nested list of SubjectMetadata to a JSON-compatible structure
    json_data = [[item.model_dump() for item in sublist] for sublist in metadata]

    # Write the JSON data to a file
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)


def extract_rebel_store(chunk_db: str):
    print(f"processing chunk db {chunk_db}")

    seen_triples = set()
    seen_entities = set()
    all_triples: List[Triple] = []
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
                valid_triples: List[Triple] = [
                    triple
                    for triple in output.triples
                    if triple.subject is not None
                       and triple.object is not None
                       and triple.predicate is not None
                ]
                unique_triples: List[Triple] = []
                for triple in valid_triples:
                    triple = triple.normalize()
                    s, p, o = (
                        triple.subject,
                        triple.predicate,
                        triple.object,
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

    print(f"found {len(all_triples)} valid triples, and {len(unique_subjects)} unique subjects")
    metadata_flat_list: List[List[SubjectMetadata]] = group_metadata_by_index([metadata for metadata_list in subject_to_metadata_mapping.values() for metadata in metadata_list])
    write_metadata_to_json(metadata_flat_list, "metadata-rebel.json")

    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    gen_kwargs = {
        "max_length": 1024,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 1,
    }

    # embeddings = get_embeddings_for_triples(all_triples, tokenizer, model)
    embeddings = get_embeddings_for_subject(unique_subjects, tokenizer, model)
    embeddings = np.array(embeddings)

    print(f"embedding shape is {embeddings.shape}")
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(1024)
    index.add(normalized_embeddings)
    faiss.write_index(index, "index-rebel.bin")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("usage : rq2-rebel.py <chunk-db.sqlite>")
    extract_rebel_store(sys.argv[1])
