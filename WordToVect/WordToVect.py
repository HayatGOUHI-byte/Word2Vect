from gensim.models import Word2Vec

# Sample corpus (list of sentences)
corpus = [
    ["this", "is", "a", "sentence", "one"],
    ["this", "is", "a", "sentence", "two"],
    ["this", "is", "a", "sentence", "three"],
    ["another", "sentence", "is", "here"],
    ["yet", "another", "sentence", "is", "here"]
]

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar("sentence")

# Print similar words
print("Similar words to 'sentence':", similar_words)
