# coding: utf-8
import json
import numpy as np
wiki_vectors = {}
import data
wiki_vocab, _, _, _ = data.get_data("/workspace/topic-preprocessing/data/wikitext/processed/full-mindf_power_law-maxdf_0.9/etm/")
nyt_vectors = {}
nyt_vocab, _, _, _ = data.get_data("/workspace/topic-preprocessing/data/nytimes/processed/full-mindf_power_law-maxdf_0.9/etm/")
wiki_emb_path = "/workspace/topic-preprocessing/data/wikitext/processed/full-mindf_power_law-maxdf_0.9/etm/embeddings.txt"
with open(wiki_emb_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word in wiki_vocab:
            vect = np.array(line[1:]).astype(np.float)
            wiki_vectors[word] = vect
wiki_embeddings = np.zeros((len(wiki_vocab), 300))
wiki_words_found = 0
for i, word in enumerate(wiki_vocab):
    try:
        wiki_embeddings[i] = wiki_vectors[word]
        wiki_words_found += 1
    except KeyError:
        wiki_embeddings[i] = np.random.normal(scale=0.6, size=(300, ))
        
print('Wiki words found = ' + str(wiki_words_found) + '/' + str(len(wiki_vocab)))
#wiki_embeddings.shape
nyt_emb_path = "/workspace/topic-preprocessing/data/nytimes/processed/full-mindf_power_law-maxdf_0.9/etm/embeddings.txt"
with open(nyt_emb_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word in nyt_vocab:
            vect = np.array(line[1:]).astype(np.float)
            nyt_vectors[word] = vect
nyt_embeddings = np.zeros((len(nyt_vocab), 300))
nyt_words_found = 0
for i, word in enumerate(nyt_vocab):
    try:
        nyt_embeddings[i] = nyt_vectors[word]
        nyt_words_found += 1
    except KeyError:
        nyt_embeddings[i] = np.random.normal(scale=0.6, size=(300, ))
        
print('NYT words found = ' + str(nyt_words_found) + '/' + str(len(nyt_vocab)))
#nyt_embeddings.shape
def get_closest_nns(word, embeddings, vocab):
    vectors = embeddings
    index = vocab.index(word)
    query = vectors[index]
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = [idx for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors
    
queries = ['political', 'china', 'basketball', 'book']
for word in queries:
    print('word: {} .. NYT neighbors: {}'.format(word, get_closest_nns(word, nyt_embeddings, nyt_vocab)))
    print('word: {} .. Wiki neighbors: {}'.format(word, get_closest_nns(word, wiki_embeddings, wiki_vocab)))
    
    
