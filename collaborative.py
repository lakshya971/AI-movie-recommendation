import os
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPED_RATINGS_PATH = os.path.join(BASE_DIR, "data", "ratings_mapped.csv")

# Global variables to cache model/ratings
_ratings_df = None
_pivot_table = None
_svd_model = None
_reconstructed_matrix = None
_movie_columns = None
_user_index_map = None

def load_and_combine_ratings():
    """
    Loads MovieLens ratings and SQLite user ratings, combined into a single DataFrame.
    """
    global _ratings_df
    
    # 1. Load MovieLens ratings
    if os.path.exists(MAPPED_RATINGS_PATH):
        try:
            movielens_ratings = pd.read_csv(MAPPED_RATINGS_PATH)
        except Exception as e:
            print(f"Error loading MovieLens ratings: {e}")
            movielens_ratings = pd.DataFrame(columns=['userId', 'tmdbId', 'rating'])
    else:
        print(f"Warning: {MAPPED_RATINGS_PATH} not found. Running collaborative filtering on SQLite ratings only.")
        movielens_ratings = pd.DataFrame(columns=['userId', 'tmdbId', 'rating'])
        
    # 2. Get ratings from database
    db_ratings_raw = []
    try:
        # Get all users from DB and their ratings
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, tmdb_id, rating FROM ratings")
        rows = cursor.fetchall()
        conn.close()
        for row in rows:
            db_ratings_raw.append({
                # Offset user ID to prevent clash with MovieLens user IDs (which are typically 1-610)
                'userId': row['user_id'] + 10000,
                'tmdbId': row['tmdb_id'],
                'rating': row['rating']
            })
    except Exception as e:
        print(f"Error reading ratings from database: {e}")
        
    db_ratings = pd.DataFrame(db_ratings_raw, columns=['userId', 'tmdbId', 'rating'])
    
    # 3. Concatenate
    if not db_ratings.empty:
        combined = pd.concat([movielens_ratings, db_ratings], ignore_index=True)
    else:
        combined = movielens_ratings
        
    # Deduplicate to prevent duplicate entries in pivot reshaping
    combined = combined.drop_duplicates(subset=['userId', 'tmdbId'], keep='last')
        
    _ratings_df = combined
    return combined

def train_collaborative_model(n_components=20):
    """
    Fits TruncatedSVD on the combined user-item rating matrix.
    """
    global _pivot_table, _svd_model, _reconstructed_matrix, _movie_columns, _user_index_map
    
    combined_ratings = load_and_combine_ratings()
    if combined_ratings.empty:
        print("No rating data available to train SVD model.")
        return False
        
    # Drop duplicates if any user rated the same movie twice
    combined_ratings = combined_ratings.drop_duplicates(subset=['userId', 'tmdbId'], keep='last')
    
    # Pivot to user-item matrix
    print("Pivoting ratings to matrix...")
    pivot_table = combined_ratings.pivot(index='userId', columns='tmdbId', values='rating').fillna(0)
    
    _pivot_table = pivot_table
    _movie_columns = pivot_table.columns.values
    _user_index_map = {user_id: idx for idx, user_id in enumerate(pivot_table.index)}
    
    n_users, n_movies = pivot_table.shape
    print(f"Rating matrix shape: {n_users} users, {n_movies} movies")
    
    # Set n_components dynamically if number of users/movies is small
    n_comp = min(n_components, n_users, n_movies)
    if n_comp < 2:
        n_comp = min(n_users, n_movies)
        if n_comp == 0:
            return False
            
    print(f"Fitting TruncatedSVD with {n_comp} components...")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    
    # Fit and transform
    matrix_dense = pivot_table.values
    transformed = svd.fit_transform(matrix_dense)
    reconstructed = svd.inverse_transform(transformed)
    
    # Cache
    _svd_model = svd
    _reconstructed_matrix = reconstructed
    print("Collaborative SVD model trained successfully.")
    return True

def recommend_for_user(user_id: int, top_n: int = 10):
    """
    Predicts ratings for a user and recommends top_n movies they haven't rated yet.
    """
    global _pivot_table, _reconstructed_matrix, _movie_columns, _user_index_map
    
    # Ensure model is trained
    if _svd_model is None or _reconstructed_matrix is None:
        success = train_collaborative_model()
        if not success:
            return []
            
    offset_user_id = user_id + 10000
    
    # Check if user is in our ratings matrix
    if offset_user_id not in _user_index_map:
        # Cold start user: Recommend popular items (items with highest mean ratings/counts)
        print(f"User {user_id} not in collaborative matrix (cold start). Recommending popular movies.")
        return get_popular_movies_fallback(top_n)
        
    user_row_idx = _user_index_map[offset_user_id]
    user_predictions = _reconstructed_matrix[user_row_idx]
    
    # Get what user has already rated
    user_ratings = _pivot_table.loc[offset_user_id]
    already_rated_indices = np.where(user_ratings > 0)[0]
    already_rated_movie_ids = set(_movie_columns[already_rated_indices])
    
    # Sort predictions
    preds_df = pd.DataFrame({
        'tmdbId': _movie_columns,
        'pred_rating': user_predictions
    })
    
    # Filter out already rated movies
    preds_df = preds_df[~preds_df['tmdbId'].isin(already_rated_movie_ids)]
    
    # Sort by predicted rating descending
    top_recs = preds_df.sort_values(by='pred_rating', ascending=False).head(top_n)
    
    return [int(x) for x in top_recs['tmdbId'].tolist()]

def get_popular_movies_fallback(top_n: int = 10):
    """
    Returns popular movies from MovieLens dataset.
    """
    global _ratings_df
    if _ratings_df is None:
        load_and_combine_ratings()
        
    if _ratings_df is None or _ratings_df.empty:
        # Return static hardcoded popular TMDB IDs as last resort
        return [299534, 19995, 24428, 299536, 155, 680, 13, 27205, 120, 122]
        
    # Group by movie and compute mean rating & count
    stats = _ratings_df.groupby('tmdbId').agg(
        mean_rating=('rating', 'mean'),
        count=('rating', 'count')
    )
    
    # Select popular items (e.g. at least 5 ratings, sorted by rating then count)
    popular = stats[stats['count'] >= 5].sort_values(by=['mean_rating', 'count'], ascending=False)
    
    if len(popular) < top_n:
        popular = stats.sort_values(by='count', ascending=False)
        
    return [int(x) for x in popular.index[:top_n].tolist()]

def get_predictions_and_actuals_for_eval():
    """
    Splits the combined ratings into train/test, fits on train, and returns predictions and actual ratings
    for evaluation metrics computing.
    """
    combined_ratings = load_and_combine_ratings()
    if combined_ratings.empty or len(combined_ratings) < 10:
        return [], []
        
    # Simple 80-20 train-test split by index shuffling
    shuffled = combined_ratings.sample(frac=1, random_state=42)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx]
    test_df = shuffled.iloc[split_idx:]
    
    # Create pivot on train
    pivot_train = train_df.pivot(index='userId', columns='tmdbId', values='rating').fillna(0)
    
    # Fit SVD
    n_users, n_movies = pivot_train.shape
    n_comp = min(15, n_users, n_movies)
    if n_comp < 1:
        return [], []
        
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    transformed = svd.fit_transform(pivot_train.values)
    reconstructed = svd.inverse_transform(transformed)
    
    train_reconstructed_df = pd.DataFrame(
        reconstructed,
        index=pivot_train.index,
        columns=pivot_train.columns
    )
    
    predictions = []
    actuals = []
    
    # Collect predictions for test set
    for idx, row in test_df.iterrows():
        u = row['userId']
        m = row['tmdbId']
        actual = row['rating']
        
        # If user and movie are in train set, retrieve prediction
        if u in train_reconstructed_df.index and m in train_reconstructed_df.columns:
            pred = train_reconstructed_df.loc[u, m]
            # Clip predictions between 0.5 and 5.0
            pred = max(0.5, min(5.0, pred))
            predictions.append(pred)
            actuals.append(actual)
            
    return predictions, actuals
