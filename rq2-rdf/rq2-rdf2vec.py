from rdf2vec.graphs import KnowledgeGraph
from rdf2vec.walkers import RandomWalker
from rdf2vec.embedders import Word2Vec
from rdf2vec.rdf2vec import RDF2VecTransformer
from rdflib import URIRef, Literal

triples = [
    (URIRef("http://example.org/Person/Alice"), URIRef("http://example.org/knows"), URIRef("http://example.org/Person/Bob")),
    (URIRef("http://example.org/Person/Alice"), URIRef("http://example.org/role"), Literal("immigration activist")),
    (URIRef("http://example.org/Person/Bob"), URIRef("http://example.org/associatedWith"), URIRef("http://example.org/Org/DisinfoInc")),
]

kg = KnowledgeGraph()
kg._triples.clear()  # clear default content

for s, p, o in triples:
    kg.add_triple(s, p, o)

# Prepare entities
# entities = list({s for s, _, _ in triples})
entities = list(kg._triples.keys())

# Setup and train RDF2Vec
walker = RandomWalker(depth=2, walks_per_entity=5)
embedder = Word2Vec(vector_size=50)
transformer = RDF2VecTransformer(walkers=[walker], embedder=embedder)

# this is a dict[URI] -> vector
embeddings = transformer.fit_transform(kg, entities)

# Map embeddings back to entities
for uri, vector in zip(entities, embeddings):
    print(f"{uri} → {vector[:5]}...")  # Preview vector
