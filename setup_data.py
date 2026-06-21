import os
import zipfile
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MOVIELENS_DIR = os.path.join(DATA_DIR, "movielens")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_movielens():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(MOVIELENS_DIR):
        print(f"Downloading MovieLens dataset from {MOVIELENS_URL}...")
        urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # Rename the extracted folder to 'movielens'
        extracted_folder = os.path.join(DATA_DIR, "ml-latest-small")
        if os.path.exists(extracted_folder):
            os.rename(extracted_folder, MOVIELENS_DIR)
            
        # Clean up zip
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        print("MovieLens dataset downloaded and extracted successfully.")
    else:
        print("MovieLens dataset already exists.")

def map_ratings_to_tmdb():
    print("Mapping MovieLens ratings to TMDB IDs...")
    ratings_path = os.path.join(MOVIELENS_DIR, "ratings.csv")
    links_path = os.path.join(MOVIELENS_DIR, "links.csv")
    
    if not os.path.exists(ratings_path) or not os.path.exists(links_path):
        raise FileNotFoundError("MovieLens ratings.csv or links.csv missing. Run download first.")
        
    ratings = pd.read_csv(ratings_path)
    links = pd.read_csv(links_path)
    
    # Drop rows where tmdbId is missing
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)
    links['movieId'] = links['movieId'].astype(int)
    
    # Merge ratings with links
    merged = pd.merge(ratings, links, on='movieId', how='inner')
    
    # Select columns needed
    mapped_ratings = merged[['userId', 'tmdbId', 'rating']]
    
    output_path = os.path.join(DATA_DIR, "ratings_mapped.csv")
    mapped_ratings.to_csv(output_path, index=False)
    print(f"Mapped ratings saved to {output_path}. Total records: {len(mapped_ratings)}")

def precompute_sentence_embeddings():
    """
    Attempts to precompute sentence embeddings using sentence-transformers.
    If sentence-transformers is not installed or errors, it falls back to a warning.
    This creates data/sentence_embeddings.pkl
    """
    df_path = os.path.join(BASE_DIR, "df.pkl")
    embeddings_output_path = os.path.join(DATA_DIR, "sentence_embeddings.pkl")
    
    if os.path.exists(embeddings_output_path):
        print("Sentence embeddings already precomputed.")
        return
        
    if not os.path.exists(df_path):
        print(f"Warning: df.pkl not found at {df_path}. Cannot compute embeddings.")
        return
        
    try:
        import pickle
        from sentence_transformers import SentenceTransformer
        
        print("Loading local movie dataset...")
        with open(df_path, "rb") as f:
            df = pickle.load(f)
            
        print("Loading sentence-transformer model ('all-MiniLM-L6-v2')...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Fill missing overviews
        overviews = df['overview'].fillna('').tolist()
        
        print(f"Computing embeddings for {len(overviews)} movies. This may take a minute...")
        embeddings = model.encode(overviews, show_progress_bar=True, batch_size=64)
        
        # Save embeddings and indices/titles map
        data_to_save = {
            "embeddings": embeddings,
            "titles": df['title'].tolist()
        }
        
        with open(embeddings_output_path, "wb") as f:
            pickle.dump(data_to_save, f)
            
        print(f"Sentence embeddings precomputed and saved to {embeddings_output_path}.")
        
    except ImportError:
        print("Sentence-transformers not installed. Skipping sentence embedding precomputation.")
    except Exception as e:
        print(f"Error during sentence embedding precomputation: {e}")

if __name__ == "__main__":
    download_movielens()
    map_ratings_to_tmdb()
    precompute_sentence_embeddings()
