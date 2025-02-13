import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example documents
documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and AI concerned with interactions between computers and human language.",
]

documents_nl = [
    "Machine learning is een data analyse methode voor het geautomatiseerd bouwen van analytische modellen.",
    "Kunstmatige intelligentie (AI) is intelligence die zichtbaar is bij machines, in contrast met natuurlijke intelligentie die zichtbaar is bij mensen en dieren.",
    "Deep learning is deel van een generiekere familie van machine learning methodes die gebaseerd zijn op kunstmatige neurale netwerken.",
    "Natural language processing (NLP) is een deel van linguistiek, computerwetenschappen, en kunstmatige intelligentie die zich bezighoudt met interacties tussen computer en menselijke taal.",
]

# Function to chunk text with overlap
def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Apply chunking to documents
chunked_documents = []
for doc in documents_nl:
    chunked_documents.extend(chunk_text(doc, chunk_size=10, overlap=3))

# Generate embeddings and create FAISS index
index_file = "/project/home/p200769/faiss_index.bin"
if os.path.exists(index_file):
    print("Loading existing FAISS index...")
    index = faiss.read_index(index_file)
else:
    print("Creating new FAISS index...")
    embeddings = model.encode(chunked_documents, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    print("Index saved to disk.")

# Example query
query = "How do machines learn from data?"
query_nl = "Hoe leren machines van data?"
query_embedding = model.encode([query_nl], convert_to_numpy=True)

# Search for top-2 similar chunks
k = 2
D, I = index.search(query_embedding, k)

# Print results
print("Query:", query_nl)
print(f"D is {D}")
print("Top matching chunks:")
for idx in I[0]:
    print("-", chunked_documents[idx])

from transformers import MarianMTModel, MarianTokenizer
import torch

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
# Force CPU execution
device = torch.device("cpu")
model.to("cpu")
def translate(text):
    print(f"generating inputs")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(f"generating translation")
    translated = model.generate(**inputs)
    print(f"batch decode")
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

print(translate("Hallo, hoe gaat het?"))

# # French → English
# print(translate("Bonjour, comment ça va?", "Helsinki-NLP/opus-mt-fr-en"))
