from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter
import torch

def extract_triplets(text):
    print(f"received text : {text}")
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'subject': subject.strip(), 'predicate': relation.strip(),'object': object_.strip()})
    return triplets

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
gen_kwargs = {
    "max_length": 1024,
    "length_penalty": 0,
    "num_beams": 3,
    "num_return_sequences": 1,
}

# Text to extract triplets from
text = """
On July 27, the Commission informed the city council about the reception of refugees who will be received in temporary emergency housing at the Prof. dr. E.M. Meijerslaan.
On July 20, 2022, State Secretary Eric van der Burg indicated that the cabinet wants to allow municipalities that they do have to compensate for Ukrainians, but not asylum seekers from other countries.
Wil Roode, Citizen Council member SP: "When on July 29, 2022, the Human Rights League brought out a statement that the cabinet thus legitimizes an unequal treatment of refugees and asylum seekers, I immediately thought that the College van Amstelveen has already done this and very quickly, namely very shortly after the ruling of 20 July. Because on July 25, refugees from the Ukraine have already been placed in the temporary emergency shelter."
Such a preferred policy is discriminatory and undesirable.
In the Netherlands there are four thousand municipal beds for the reception of Ukrainians unoccupied, while in Ter Apel there are regularly asylum seekers in the grass.
With policy only to offer care to people from the Ukraine, a municipality immediately distinguishes based on origin between people who flee Ukraine and asylum seekers from other countries.
Wil Roode: "Not only there are already 26 selected families from the Ukraine in the emergency shelter, but the college has used selection criteria without submitting it to the city council.
The SP wants to know why selections have taken place and whether these criteria have been tested. On our own selecting we find incorrect."
The SP therefore asks written questions for clarification about this issue because the Commission runs the risk because of selecting on the basis of origin to be guilty of unequal treatment.
"""

# text = 'Punta Cana is a resort town in the municipality of Higüey, in La Altagracia Province, the easternmost province of the Dominican Republic.'


MAX_WORDS = 200  # Chunk size in words
OVERLAP = 10  # Overlapping words to maintain context

# Initialize the LlamaIndex Sentence Splitter
splitter = SentenceSplitter(chunk_size=MAX_WORDS, chunk_overlap=OVERLAP)
chunks = splitter.split_text(text)

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
def get_embeddings_for_triples(triples):
    embeddings = []

    for triple in triples:
        # Create textual representation of the triple
        text = f"{triple['subject']} {triple['predicate']} {triple['object']}"

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

# Example triples
triples = [
    {"subject": "Wil Roode", "predicate": "stated", "object": "refugees"},
    {"subject": "Wil Roode", "predicate": "stated", "object": "housing"},
    {"subject": "SP", "predicate": "wants", "object": "clarification"}
]

# model.eval()
# Get embeddings for the triples
embeddings = get_embeddings_for_triples(triples)

# Print the embeddings
for idx, embedding in enumerate(embeddings):
    print(f"Embedding for triple {idx+1} with dimension {embedding.shape}:")
    print(embedding)
