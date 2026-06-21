import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import custom modules
import database
import collaborative
import sentence_rec
import knowledge
import chatbot
import evaluation

# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="AI Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# PYDANTIC SCHEMAS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []
    original_language: Optional[str] = None


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


class AuthRequest(BaseModel):
    username: str
    password: str


class RateRequest(BaseModel):
    tmdb_id: int
    rating: float
    review_text: str = ""


class HistoryRequest(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None


class WishlistRequest(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    current_movie_context: Optional[str] = None


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
        original_language=data.get("original_language"),
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl format invalid")


def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD RESOURCES
# =========================
@app.on_event("startup")
def load_resources():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    # Load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    # Load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    # Load TF-IDF matrix
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Load tfidf vectorizer
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    # Build normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    # Init SQLite database
    database.init_db()

    # Pre-train Collaborative Filtering SVD model
    try:
        collaborative.train_collaborative_model()
    except Exception as e:
        print(f"Error pre-training collaborative SVD model: {e}")


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- AUTH ROUTES ----------
@app.post("/auth/register")
def register(payload: AuthRequest):
    try:
        uid = database.register_user(payload.username, payload.password)
        return {"status": "success", "user_id": uid, "username": payload.username}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login")
def login(payload: AuthRequest):
    user = database.authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"status": "success", "user_id": user["id"], "username": user["username"]}


# ---------- USER DATA ROUTES ----------
@app.post("/user/{uid}/rate")
def rate_movie(uid: int, payload: RateRequest):
    database.add_rating(uid, payload.tmdb_id, payload.rating, payload.review_text)
    # Retrain model dynamically so recommendations update immediately!
    try:
        collaborative.train_collaborative_model()
    except Exception as e:
        print(f"Error retraining collaborative model: {e}")
    return {"status": "success"}


@app.get("/user/{uid}/ratings")
def get_ratings(uid: int):
    return database.get_user_ratings(uid)


@app.post("/user/{uid}/history")
def add_history(uid: int, payload: HistoryRequest):
    database.add_to_history(uid, payload.tmdb_id, payload.title, payload.poster_url or "")
    return {"status": "success"}


@app.get("/user/{uid}/history")
def get_history_route(uid: int):
    return database.get_history(uid)


@app.post("/user/{uid}/wishlist")
def toggle_wishlist_route(uid: int, payload: WishlistRequest):
    added = database.toggle_wishlist(uid, payload.tmdb_id, payload.title, payload.poster_url or "")
    return {"status": "success", "added": added}


@app.get("/user/{uid}/wishlist")
def get_wishlist_route(uid: int):
    return database.get_wishlist(uid)


@app.get("/user/{uid}/wishlist/check")
def check_wishlist(uid: int, tmdb_id: int):
    status = database.check_wishlist_status(uid, tmdb_id)
    return {"in_wishlist": status}


# ---------- RECOMMENDATION ROUTES ----------

@app.get("/recommend/collaborative", response_model=List[TMDBMovieCard])
async def recommend_collab(user_id: int = Query(...), top_n: int = Query(10)):
    tmdb_ids = collaborative.recommend_for_user(user_id, top_n)
    cards = []
    for tid in tmdb_ids:
        try:
            details = await tmdb_movie_details(tid)
            cards.append(
                TMDBMovieCard(
                    tmdb_id=tid,
                    title=details.title,
                    poster_url=details.poster_url,
                    release_date=details.release_date,
                    vote_average=details.vote_average,
                )
            )
        except Exception:
            continue
    return cards


@app.get("/recommend/sentence")
async def recommend_sentence_endpoint(title: str = Query(...), top_n: int = Query(10)):
    recs = sentence_rec.recommend_by_embedding(title, top_n)
    
    # If no precomputed embeddings are loaded, fallback to TF-IDF
    if not recs:
        try:
            tfidf_recs = tfidf_recommend_titles(title, top_n=top_n)
            recs = [(t, s) for t, s in tfidf_recs]
        except Exception:
            recs = []
            
    # Attach TMDB details
    out = []
    for t, s in recs:
        card = await attach_tmdb_card_by_title(t)
        out.append({"title": t, "score": s, "tmdb": card})
    return out


@app.get("/recommend/knowledge", response_model=List[TMDBMovieCard])
def recommend_knowledge_route(
    user_id: Optional[int] = Query(None),
    current_tmdb_id: Optional[int] = Query(None),
    top_n: int = Query(10)
):
    recs = knowledge.recommend_knowledge(user_id, current_tmdb_id, top_n)
    return [TMDBMovieCard(**r) for r in recs]


@app.get("/recommend/hybrid", response_model=List[TMDBMovieCard])
async def recommend_hybrid(
    title: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    current_tmdb_id: Optional[int] = Query(None),
    top_n: int = Query(12)
):
    """
    Blends recommendations:
    - 40% Content TF-IDF
    - 20% Content Sentence Transformer
    - 30% Collaborative Filtering
    - 10% Knowledge-Based
    """
    candidates = {}  # tmdb_id -> {title, tfidf, sentence, collab, knowledge}

    # Helper to add score to candidates
    def add_score(tmdb_id: int, movie_title: str, score_type: str, val: float):
        if tmdb_id not in candidates:
            candidates[tmdb_id] = {
                "title": movie_title,
                "tfidf": 0.0,
                "sentence": 0.0,
                "collab": 0.0,
                "knowledge": 0.0,
            }
        candidates[tmdb_id][score_type] = val

    # 1. Fetch TF-IDF recommendations
    if title:
        try:
            tfidf_recs = tfidf_recommend_titles(title, top_n=30)
            for m_title, score in tfidf_recs:
                card = await attach_tmdb_card_by_title(m_title)
                if card:
                    add_score(card.tmdb_id, card.title, "tfidf", score)
        except Exception:
            pass

    # 2. Fetch Sentence Transformer recommendations
    if title:
        try:
            sent_recs = sentence_rec.recommend_by_embedding(title, top_n=30)
            for m_title, score in sent_recs:
                card = await attach_tmdb_card_by_title(m_title)
                if card:
                    add_score(card.tmdb_id, card.title, "sentence", score)
        except Exception:
            pass

    # 3. Fetch Collaborative Filtering SVD predicted ratings
    if user_id:
        try:
            # We can get predictions for the candidate list, or generate top collab recs
            collab_ids = collaborative.recommend_for_user(user_id, top_n=30)
            for idx, tid in enumerate(collab_ids):
                # Score degrades linearly by rank (from 1.0 down to 0.1)
                score = 1.0 - (idx * 0.03)
                try:
                    details = await tmdb_movie_details(tid)
                    add_score(tid, details.title, "collab", max(0.1, score))
                except Exception:
                    continue
        except Exception:
            pass

    # 4. Fetch Knowledge-Based recommendations
    try:
        know_recs = knowledge.recommend_knowledge(user_id, current_tmdb_id, top_n=30)
        for idx, r in enumerate(know_recs):
            score = 1.0 - (idx * 0.03)
            add_score(r["tmdb_id"], r["title"], "knowledge", max(0.1, score))
    except Exception:
        pass

    # Calculate blended hybrid score and build cards
    hybrid_list = []
    for tmdb_id, scores in candidates.items():
        if current_tmdb_id and tmdb_id == current_tmdb_id:
            continue
            
        h_score = (
            0.40 * scores["tfidf"]
            + 0.20 * scores["sentence"]
            + 0.30 * scores["collab"]
            + 0.10 * scores["knowledge"]
        )
        hybrid_list.append((tmdb_id, scores["title"], h_score))

    # Sort candidates by hybrid score
    hybrid_list.sort(key=lambda x: x[2], reverse=True)

    # Build final cards
    out = []
    for tmdb_id, movie_title, _ in hybrid_list[:top_n]:
        try:
            details = await tmdb_movie_details(tmdb_id)
            out.append(
                TMDBMovieCard(
                    tmdb_id=tmdb_id,
                    title=details.title,
                    poster_url=details.poster_url,
                    release_date=details.release_date,
                    vote_average=details.vote_average,
                )
            )
        except Exception:
            continue
            
    # Fallback to general discover if hybrid list is empty
    if not out:
        discover = await tmdb_get("/movie/popular", {"language": "en-US", "page": 1})
        out = await tmdb_cards_from_results(discover.get("results", []), limit=top_n)

    return out


@app.get("/recommend/mood", response_model=List[TMDBMovieCard])
async def recommend_mood_route(mood: str = Query(...), limit: int = Query(18)):
    mood_norm = mood.strip().lower()
    genre_ids = chatbot.MOOD_GENRE_MAPPING.get(mood_norm, [35]) # default to Comedy
    
    # Randomly pick first genre ID and fetch discover results
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": ",".join(map(str, genre_ids)),
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return cards


# ---------- AI FEATURES ROUTES ----------

@app.get("/explain")
def get_explanation(
    title: str = Query(...),
    rec_type: str = Query(...),
    details: Optional[str] = Query("")
):
    explanation = chatbot.explain_recommendation(title, rec_type, details)
    return {"explanation": explanation}


@app.post("/chatbot")
def run_chatbot(payload: ChatRequest):
    response = chatbot.chat_with_llm(
        payload.message,
        payload.history,
        payload.current_movie_context
    )
    return {"response": response}


# ---------- METRICS & EVALUATION ROUTE ----------
@app.get("/evaluate/metrics")
def get_evaluation_metrics():
    # Return evaluations calculated dynamically on MovieLens small test split
    try:
        metrics = evaluation.calculate_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics evaluation failed: {e}")


# ---------- TMDB UTILS ROUTES ----------
@app.get("/movie/{movie_id}/videos")
async def get_movie_videos(movie_id: int):
    # Fetch trailers and clips from TMDB
    data = await tmdb_get(f"/movie/{movie_id}/videos", {"language": "en-US"})
    return data


@app.get("/movie/{movie_id}/credits")
async def get_movie_credits(movie_id: int):
    # Fetch cast and crew details from TMDB
    data = await tmdb_get(f"/movie/{movie_id}/credits", {"language": "en-US"})
    return data


@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/home/discover", response_model=List[TMDBMovieCard])
async def discover_movies(
    language: Optional[str] = Query(None),
    genre: Optional[int] = Query(None),
    limit: int = Query(24)
):
    params = {
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": 1
    }
    if language:
        params["with_original_language"] = language
    if genre:
        params["with_genres"] = genre
        
    discover = await tmdb_get("/discover/movie", params)
    return await tmdb_cards_from_results(discover.get("results", []), limit=limit)


# ---------- SEARCH BUNDLE (Details + Hybrid recs + Genre recs) ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


@app.get("/tmdb/search")
async def tmdb_search_endpoint(query: str = Query(...), page: int = Query(1)):
    return await tmdb_search_movies(query, page)


@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_by_genre(tmdb_id: int = Query(...), limit: int = Query(12)):
    try:
        details = await tmdb_movie_details(tmdb_id)
        if details.genres:
            genre_id = details.genres[0]["id"]
            discover = await tmdb_get(
                "/discover/movie",
                {
                    "with_genres": genre_id,
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "page": 1,
                },
            )
            cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit + 1)
            return [c for c in cards if c.tmdb_id != tmdb_id][:limit]
    except Exception as e:
        print(f"Error in recommend_by_genre: {e}")
        
    discover = await tmdb_get("/movie/popular", {"language": "en-US", "page": 1})
    return await tmdb_cards_from_results(discover.get("results", []), limit=limit)

