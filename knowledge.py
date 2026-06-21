import os
import requests
from dotenv import load_dotenv
import database

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_api_call(path: str, params: dict) -> dict:
    if not TMDB_API_KEY:
        return {}
    p = dict(params)
    p["api_key"] = TMDB_API_KEY
    try:
        r = requests.get(f"{TMDB_BASE}{path}", params=p, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"TMDB request in knowledge engine failed: {e}")
    return {}

def get_movie_details(tmdb_id: int) -> dict:
    return tmdb_api_call(f"/movie/{tmdb_id}", {"language": "en-US"})

def get_movie_credits(tmdb_id: int) -> dict:
    return tmdb_api_call(f"/movie/{tmdb_id}/credits", {})

def recommend_knowledge(user_id: int, current_tmdb_id: int = None, top_n: int = 10):
    """
    Knowledge-Based Recommendation Engine using explicit rules:
    Rule 1: Genre Preference - Recommend highly-rated genres from history.
    Rule 2: Director/Cast Match - If a user rated movies with director X high, recommend director X's other popular movies.
    Rule 3: Similarity Fallback - If no ratings, recommend top-rated movies of the same genre as current_tmdb_id.
    """
    ratings = database.get_user_ratings(user_id) if user_id else []
    
    # Filter highly rated movies (rating >= 4.0)
    high_ratings = [r for r in ratings if r["rating"] >= 4.0]
    
    preferred_genres = {}
    preferred_directors = {}
    rated_movie_ids = {r["tmdb_id"] for r in ratings}
    if current_tmdb_id:
        rated_movie_ids.add(current_tmdb_id)
        
    # Analyze preferences
    for r in high_ratings[:10]: # Check last 10 highly rated movies
        m_details = get_movie_details(r["tmdb_id"])
        if m_details:
            for g in m_details.get("genres", []):
                g_id = g["id"]
                preferred_genres[g_id] = preferred_genres.get(g_id, 0) + 1
                
            # Fetch director details
            credits = get_movie_credits(r["tmdb_id"])
            if credits:
                for crew_member in credits.get("crew", []):
                    if crew_member.get("job") == "Director":
                        d_id = crew_member["id"]
                        preferred_directors[d_id] = preferred_directors.get(d_id, 0) + 1
                        break
                        
    # Sort preferences
    sorted_genres = sorted(preferred_genres.items(), key=lambda x: x[1], reverse=True)
    sorted_directors = sorted(preferred_directors.items(), key=lambda x: x[1], reverse=True)
    
    recommended_movies = []
    
    # Scenario A: User has genre/director preferences
    if sorted_genres:
        top_genre_id = sorted_genres[0][0]
        
        # Rule 1: Highly rated Genre
        discover_params = {
            "with_genres": top_genre_id,
            "sort_by": "popularity.desc",
            "language": "en-US",
            "page": 1
        }
        
        # Rule 2: If director exists, prioritize discover with director
        if sorted_directors:
            top_director_id = sorted_directors[0][0]
            discover_params["with_crew"] = top_director_id
            
        data = tmdb_api_call("/discover/movie", discover_params)
        results = data.get("results", [])
        
        # If no results (e.g. director has no other movies in that genre), try genre only
        if not results and "with_crew" in discover_params:
            del discover_params["with_crew"]
            data = tmdb_api_call("/discover/movie", discover_params)
            results = data.get("results", [])
            
        for m in results:
            m_id = m["id"]
            if m_id not in rated_movie_ids:
                recommended_movies.append(m)
                
    # Scenario B: No user history, but current movie details are provided (Rule 3)
    elif current_tmdb_id:
        m_details = get_movie_details(current_tmdb_id)
        if m_details and m_details.get("genres"):
            genre_id = m_details["genres"][0]["id"]
            data = tmdb_api_call("/discover/movie", {
                "with_genres": genre_id,
                "sort_by": "vote_average.desc",
                "vote_count.gte": 500, # ensure high rating is quality, not single vote
                "language": "en-US",
                "page": 1
            })
            for m in data.get("results", []):
                m_id = m["id"]
                if m_id not in rated_movie_ids:
                    recommended_movies.append(m)
                    
    # Scenario C: Complete cold start, no current movie -> popular movies
    if not recommended_movies:
        data = tmdb_api_call("/movie/popular", {"language": "en-US", "page": 1})
        for m in data.get("results", []):
            m_id = m["id"]
            if m_id not in rated_movie_ids:
                recommended_movies.append(m)
                
    # Parse to simple list of dicts/cards matching TMDBMovieCard schema
    out = []
    for m in recommended_movies[:top_n]:
        out.append({
            "tmdb_id": int(m["id"]),
            "title": m.get("title") or m.get("name") or "",
            "poster_url": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
            "release_date": m.get("release_date"),
            "vote_average": m.get("vote_average")
        })
        
    return out
