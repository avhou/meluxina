from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from rdflib import Literal
from pyrdf2vec.samplers import PageRankSampler


from pyrdf2vec.graphs import KG, Vertex

triples = [
    ["Alice", "knows", "Bob"],
    ["Alice", "knows", "Dean"],
    ["Dean", "loves", "Alice"],
]

kg = KG()

for s, p, o in triples:
    subj = Vertex(s)
    obj = Vertex(o)
    pred = Vertex(p, predicate=True, vprev=subj, vnext=obj)
    kg.add_walk(subj, pred, obj)


unique_subjects = list({s for s, _, _ in triples})
unique_objects = list({o for _, _, o in triples})
unique_subjects_and_objects = list(set(unique_subjects + unique_objects))
# we willen in principe kunnen queryen op de subjects
entities = unique_subjects

# # Setup and train RDF2Vec
walker = RandomWalker(4, 10, PageRankSampler())
embedder = Word2Vec(epochs=10)
transformer = RDF2VecTransformer(embedder, walkers=[walker], verbose=1)
#
# embeddings is een lijst van vectors, de size van de gevraagde entiteiten.  dus elke entiteit heeft een embedding vector
embeddings, _ = transformer.fit_transform(kg, entities)
#
print("embeddings : ")
print(len(embeddings))
