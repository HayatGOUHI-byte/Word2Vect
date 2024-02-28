from gensim.models import KeyedVectors

# Charger le modèle Word2Vec pré-entraîné
model_path = "GoogleNews-vectors-negative300.bin.gz"
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=500000)  # Chargez seulement les premiers 500 000 mots pour cet exemple

# Obtenir le vecteur correspondant au mot "car"
word = "car"
if word in word_vectors:
    word_vector = word_vectors[word]
    print(f"Vecteur pour le mot '{word}': {word_vector}")
else:
    print(f"Le mot '{word}' n'est pas présent dans le modèle Word2Vec.")
