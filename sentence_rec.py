import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "sentence_embeddings.pkl")

# Cached embeddings
_embeddings_data = None

def load_embeddings():
    global _embeddings_data
    if _embeddings_data is not None:
        return _embeddings_data
        
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Sentence embeddings file {EMBEDDINGS_PATH} not found. Semantic search fallback to TF-IDF.")
        return None
        
    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            _embeddings_data = pickle.load(f)
        print("Sentence embeddings loaded successfully.")
        return _embeddings_data
    except Exception as e:
        print(f"Error loading sentence embeddings: {e}")
        return None

def recommend_by_embedding(title: str, top_n: int = 10):
    """
    Computes semantic similarity based on sentence transformer embeddings.
    """
    data = load_embeddings()
    if not data:
        return []
        
    embeddings = data.get("embeddings")
    titles = data.get("titles")
    
    if embeddings is None or titles is None:
        return []
        
    # Find matching movie index
    norm_titles = [str(t).strip().lower() for t in titles]
    query_norm = str(title).strip().lower()
    
    if query_norm not in norm_titles:
        print(f"Title '{title}' not found in sentence embeddings title map.")
        return []
        
    idx = norm_titles.index(query_norm)
    
    # Calculate cosine similarity
    query_vector = embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(query_vector, embeddings)[0]
    
    # Sort indexes descending by score
    sorted_indices = np.argsort(-similarities)
    
    recs = []
    for i in sorted_indices:
        if i == idx:
            continue
        recs.append((titles[i], float(similarities[i])))
        if len(recs) >= top_n:
            break
            
    return recs
