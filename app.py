import os
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =============================
# CONFIG & STATE
# =============================
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(page_title="VibeRec — AI Movie Recommender", page_icon="🍿", layout="wide")

# Session state initialization
if "view" not in st.session_state:
    st.session_state.view = "home"  # home | details
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Handle voice search redirect query param
voice_query = st.query_params.get("voice_search")
if voice_query:
    st.session_state.search_query = voice_query
    # Clear query param to prevent loop
    st.query_params.clear()

# Handle details redirection
qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")
if qp_view == "details" and qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except:
        pass

# =============================
# AESTHETICS & CUSTOM CSS
# =============================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Movie Card styling */
    .movie-card {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 12px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    
    .movie-card:hover {
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.15);
    }
    
    .movie-card img {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    
    .movie-title {
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.25rem;
        height: 2.5rem;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        color: #f1f5f9;
        margin-top: 6px;
    }
    
    .movie-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Info Card Panel */
    .info-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        transform: scale(1.02);
    }
    
    /* Evaluation metrics cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
        margin-top: 8px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Mood button styling */
    .mood-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .mood-btn:hover {
        background: rgba(56, 189, 248, 0.1);
        border-color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# ROUTING & NAVIGATION
# =============================
def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()

def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()

# =============================
# API REQUEST HELPERS
# =============================
def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"

def api_post(path: str, payload: dict | None = None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        return r.json(), None
    except Exception as e:
        return None, f"Request failed: {e}"

# =============================
# VOICE SEARCH SPEECH REC
# =============================
def render_voice_button():
    js_code = """
    <div style="text-align: center; padding: 10px;">
        <button id="voice-btn" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; border: none; padding: 10px 22px; border-radius: 24px; font-weight: 600; cursor: pointer; font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3); transition: all 0.2s;">
            🎤 Speak to Search
        </button>
        <p id="voice-status" style="font-size: 0.8rem; color: #94a3b8; margin-top: 6px; font-family: 'Segoe UI', Roboto, sans-serif;">Click and say a movie name...</p>
    </div>
    <script>
    const btn = document.getElementById('voice-btn');
    const status = document.getElementById('voice-status');
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        status.textContent = 'Voice recognition not supported in this browser. Try Chrome.';
        btn.disabled = true;
        btn.style.opacity = 0.5;
    } else {
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        btn.onclick = () => {
            try {
                recognition.start();
                btn.style.background = '#b91c1c';
                btn.textContent = '🛑 Listening...';
                status.textContent = 'Listening for a movie title...';
            } catch (e) {
                status.textContent = 'Already listening or error: ' + e.message;
            }
        };
        
        recognition.onresult = (event) => {
            const speechToText = event.results[0][0].transcript;
            status.textContent = 'Searching for: "' + speechToText + '"';
            
            // Redirect parent page with search term
            const url = new URL(window.parent.location.href);
            url.searchParams.set('voice_search', speechToText);
            window.parent.location.href = url.toString();
        };
        
        recognition.onerror = (event) => {
            btn.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
            btn.textContent = '🎤 Speak to Search';
            status.textContent = 'Speech error: ' + event.error;
        };
        
        recognition.onspeechend = () => {
            recognition.stop();
            btn.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
            btn.textContent = '🎤 Speak to Search';
        };
    }
    </script>
    """
    components.html(js_code, height=95)

# =============================
# REUSABLE POSTER GRID
# =============================
def render_movie_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies found matching these criteria.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0
    for r in range(rows):
        colset = st.columns(cols)
        for c in range(cols):
            if idx >= len(cards):
                break
            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id") or m.get("id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")
            vote = m.get("vote_average", 0.0)
            date = m.get("release_date") or ""
            year = date[:4] if len(date) >= 4 else ""

            with colset[c]:
                # Wrap in CSS card class using markdown
                st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
                if poster:
                    st.image(poster, width="stretch")
                else:
                    st.write("🖼️ No Poster Available")
                
                st.markdown(f"<div class='movie-title'>{title}</div>", unsafe_allow_html=True)
                
                meta_str = f"⭐ {vote:.1f}" if vote else "No Ratings"
                if year:
                    meta_str += f" | {year}"
                st.markdown(f"<div class='movie-meta'>{meta_str}</div>", unsafe_allow_html=True)
                
                if st.button("Open Details", key=f"{key_prefix}_{r}_{c}_{idx}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)
                st.markdown("</div>", unsafe_allow_html=True)

# =============================
# SIDEBAR NAVIGATION & LOGIN
# =============================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #38bdf8; font-weight: 700;'>🍿 VIBEREC</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Next-Gen AI Movie Recommender</p>", unsafe_allow_html=True)
    st.divider()
    
    # 1. Login / Register Section
    if st.session_state.logged_in:
        st.success(f"Logged in as: **{st.session_state.username}**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.subheader("🔑 Member Login")
        auth_action = st.radio("Choose Action", ["Login", "Register"], horizontal=True)
        username = st.text_input("Username", key="auth_username")
        password = st.text_input("Password", type="password", key="auth_password")
        
        if st.button("Submit Credentials", use_container_width=True):
            if not username or not password:
                st.warning("Please enter both username and password.")
            else:
                if auth_action == "Register":
                    data, err = api_post("/auth/register", {"username": username, "password": password})
                    if err:
                        st.error(f"Registration failed: {err}")
                    else:
                        st.success("Registered successfully! Please Login.")
                else:
                    data, err = api_post("/auth/login", {"username": username, "password": password})
                    if err:
                        st.error(f"Login failed: {err}")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.user_id = data["user_id"]
                        st.session_state.username = data["username"]
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                        
    st.divider()
    
    # 2. Main Navigation Options
    st.subheader("🧭 Navigation")
    pages = [
        "🏠 Home Feed",
        "🤖 AI Chatbot Assistant",
        "🎭 Mood-based Finder",
        "❤️ My Wishlist",
        "📜 Watch History",
        "⭐ My Ratings & Reviews",
        "📊 System Evaluation"
    ]
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home Feed"
        
    selected_page = st.radio("Go to", pages, index=pages.index(st.session_state.page))
    st.session_state.page = selected_page
    
    # Grid Cols slider
    grid_cols = st.slider("Grid Layout Columns", 4, 8, 6)

# ==========================================================
# VIEW: DETAILS (Expanded Movie Details + Trailers + Reviews)
# ==========================================================
if st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id
    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home Feed"):
            goto_home()
        st.stop()

    # Back to Home
    if st.button("← Back to Home Feed"):
        goto_home()

    # Log watch history if logged in
    if st.session_state.logged_in:
        # We need movie title and poster to save history. Let's fetch details first.
        details_data, err = api_get(f"/movie/id/{tmdb_id}")
        if not err and details_data:
            api_post(f"/user/{st.session_state.user_id}/history", {
                "tmdb_id": tmdb_id,
                "title": details_data.get("title", ""),
                "poster_url": details_data.get("poster_url", "")
            })

    # Fetch details
    details, err = api_get(f"/movie/id/{tmdb_id}")
    if err or not details:
        st.error(f"Could not load movie details: {err or 'Unknown error'}")
        st.stop()

    # Layout: Poster LEFT, Details RIGHT
    left, right = st.columns([1, 2.5], gap="large")

    with left:
        st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
        if details.get("poster_url"):
            st.image(details["poster_url"], width="stretch")
        else:
            st.write("🖼️ No Poster Available")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Wishlist button (heart)
        if st.session_state.logged_in:
            wishlist_status, _ = api_get(f"/user/{st.session_state.user_id}/wishlist/check", {"tmdb_id": tmdb_id})
            in_wishlist = wishlist_status.get("in_wishlist", False) if wishlist_status else False
            
            wish_label = "❤️ In Wishlist (Click to remove)" if in_wishlist else "🖤 Add to Wishlist"
            if st.button(wish_label, use_container_width=True):
                res, _ = api_post(f"/user/{st.session_state.user_id}/wishlist", {
                    "tmdb_id": tmdb_id,
                    "title": details.get("title", ""),
                    "poster_url": details.get("poster_url", "")
                })
                st.rerun()
        else:
            st.info("💡 Login to save movies to Wishlist.")

    with right:
        st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='margin:0; color:#38bdf8;'>{details.get('title','')}</h1>", unsafe_allow_html=True)
        
        release = details.get("release_date") or "Unknown"
        genres = ", ".join([g["name"] for g in details.get("genres", [])]) or "N/A"
        vote = details.get("vote_average", 0.0)
        lang = details.get("original_language", "en").upper()
        
        st.markdown(
            f"<div style='margin-top: 5px; color:#94a3b8; font-size: 0.95rem;'>"
            f"📅 Release: <b>{release}</b> | 🎭 Genres: <b>{genres}</b> | ⭐ Rating: <b>{vote:.1f}/10</b> | 🌐 Language: <b>{lang}</b>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        st.markdown("### Synopsis")
        st.write(details.get("overview") or "No plot summary available.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Star Rating & Review Submitting
        if st.session_state.logged_in:
            st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
            st.subheader("⭐ Rate & Review Movie")
            
            # Fetch existing rating
            ratings_list, _ = api_get(f"/user/{st.session_state.user_id}/ratings")
            user_rating_val = 3.0
            user_review_val = ""
            if ratings_list:
                for r in ratings_list:
                    if r["tmdb_id"] == tmdb_id:
                        user_rating_val = float(r["rating"])
                        user_review_val = r["review_text"] or ""
                        break
            
            rating = st.slider("Your Rating", 1.0, 5.0, user_rating_val, 0.5)
            review = st.text_area("Write a Review", user_review_val, placeholder="Share your thoughts about this movie...")
            
            if st.button("Submit Rating & Review"):
                res, err_rate = api_post(f"/user/{st.session_state.user_id}/rate", {
                    "tmdb_id": tmdb_id,
                    "rating": rating,
                    "review_text": review
                })
                if not err_rate:
                    st.success("Thank you for your rating! Collaborative model updated.")
                else:
                    st.error(f"Error submitting rating: {err_rate}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💡 Login to submit ratings and reviews.")

    # Trailer Section
    st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
    st.subheader("🎬 Official Trailers & Teasers")
    video_data, err_v = api_get(f"/movie/{tmdb_id}/videos")
    if not err_v and video_data and video_data.get("results"):
        videos = video_data["results"]
        # Look for YouTube trailers first
        trailers = [v for v in videos if v["site"] == "YouTube" and v["type"] in ("Trailer", "Teaser")]
        if trailers:
            selected_trailer = trailers[0]
            st.video(f"https://www.youtube.com/watch?v={selected_trailer['key']}")
        else:
            st.write("No YouTube trailers found for this movie.")
    else:
        st.write("Trailers not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Cast & Crew Section
    st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
    st.subheader("👥 Top Cast & Credits")
    credit_data, err_c = api_get(f"/movie/{tmdb_id}/credits")
    if not err_c and credit_data and credit_data.get("cast"):
        cast_list = credit_data["cast"][:12]
        cols = st.columns(6)
        for i, actor in enumerate(cast_list):
            c_idx = i % 6
            with cols[c_idx]:
                p_path = actor.get("profile_path")
                if p_path:
                    st.image(f"https://image.tmdb.org/t/p/w185{p_path}", width="stretch")
                else:
                    st.write("👤 No Photo")
                st.markdown(f"<b>{actor['name']}</b><br><span style='font-size:0.8rem; color:#94a3b8;'>{actor['character']}</span>", unsafe_allow_html=True)
    else:
        st.write("Cast details not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Explainable AI
    st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
    st.subheader("🧠 Explainable AI — Why am I seeing this?")
    # Select which recommendation context is explaining (default Hybrid)
    rec_context = st.selectbox("Explain type", ["Hybrid Blender Recommendation", "Content-Based Semantic (Sentence Transformers)", "Collaborative SVD (Latent Factor)", "Knowledge-Based Rule Engine"])
    
    short_type = "hybrid"
    if "Sentence" in rec_context:
        short_type = "sentence"
    elif "Collaborative" in rec_context:
        short_type = "collaborative"
    elif "Knowledge" in rec_context:
        short_type = "knowledge"
        
    explanation_res, err_e = api_get("/explain", {
        "title": details.get("title", ""),
        "rec_type": short_type,
        "details": f"Target movie is {details.get('title')}. Genres: {genres}. Language: {lang}. TMDB rating: {vote:.1f}."
    })
    
    if not err_e and explanation_res:
        st.markdown(f"<blockquote style='border-left: 4px solid #38bdf8; padding-left: 15px; font-style: italic; color: #cbd5e1;'>\"{explanation_res['explanation']}\"</blockquote>", unsafe_allow_html=True)
    else:
        st.write("Could not generate explanation right now.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Hybrid Recommendations Section
    st.divider()
    st.subheader("🧬 Unified Hybrid Recommendations")
    st.write("Blended recommendations matching this movie's themes, genre rules, and your personalized tastes:")
    
    hybrid_recs, err_h = api_get("/recommend/hybrid", {
        "title": details.get("title", ""),
        "user_id": st.session_state.user_id if st.session_state.logged_in else None,
        "current_tmdb_id": tmdb_id,
        "top_n": 12
    })
    
    if not err_h and hybrid_recs:
        render_movie_grid(hybrid_recs, cols=grid_cols, key_prefix="details_hybrid")
    else:
        st.warning("Could not load hybrid recommendations. Showing fallback genre recommendation.")
        genre_fallback, err_gf = api_get("/recommend/genre", {"tmdb_id": tmdb_id, "limit": 12})
        if not err_gf and genre_fallback:
            render_movie_grid(genre_fallback, cols=grid_cols, key_prefix="details_genre_fallback")

# ==========================================================
# VIEW: HOME FEED
# ==========================================================
elif st.session_state.page == "🏠 Home Feed":
    st.markdown("<h1 style='color:#38bdf8;'>🏠 Discover Movies</h1>", unsafe_allow_html=True)
    
    # Search controls layout
    s_col1, s_col2 = st.columns([3, 1])
    
    with s_col1:
        typed = st.text_input(
            "Search movie by title", 
            value=st.session_state.search_query, 
            placeholder="Type query (e.g. Inception, Avengers, Batman) and press Enter...",
            key="home_search_input"
        )
        st.session_state.search_query = typed
        
    with s_col2:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        render_voice_button()
        
    # Filters sidebar/layout
    st.markdown("### 🔍 Filters")
    f_col1, f_col2, f_col3 = st.columns([1, 1, 1.5])
    
    with f_col1:
        lang_filter = st.selectbox(
            "Language",
            ["All", "English (en)", "Hindi (hi)", "Spanish (es)", "French (fr)", "Korean (ko)", "Japanese (ja)"],
            index=0
        )
    with f_col2:
        # Basic genres mapping
        GENRE_MAP = {
            "All": None, "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, 
            "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14, 
            "History": 36, "Horror": 27, "Romance": 10749, "Sci-Fi": 878, "Thriller": 53
        }
        genre_filter = st.selectbox("Genre Filter", list(GENRE_MAP.keys()), index=0)
        
    with f_col3:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        show_personalized = False
        if st.session_state.logged_in:
            show_personalized = st.checkbox("🔮 Show My Personalized Feed (SVD Collaborative Filtering)")
        else:
            st.caption("🔒 Log in to enable personalized SVD Collaborative Filtering feed!")

    st.divider()

    # Search Mode Execution
    if st.session_state.search_query.strip():
        q_term = st.session_state.search_query.strip()
        st.markdown(f"### 🔎 Search Results for: \"{q_term}\"")
        
        # Clear search button
        if st.button("Clear Search"):
            st.session_state.search_query = ""
            st.query_params.clear()
            st.rerun()
            
        data, err = api_get("/tmdb/search", {"query": q_term})
        if err or not data:
            st.error(f"Search failed: {err}")
        else:
            # Parse search results
            results = data.get("results", []) if data else []
            cards = []
            for m in results[:24]:
                cards.append({
                    "tmdb_id": int(m["id"]),
                    "title": m.get("title") or m.get("name") or "",
                    "poster_url": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
                    "release_date": m.get("release_date"),
                    "vote_average": m.get("vote_average", 0.0)
                })
            if cards:
                render_movie_grid(cards, cols=grid_cols, key_prefix="search_results")
            else:
                st.info("No matching movies found on TMDB.")
                
    # Personalized Collaborative Feed
    elif show_personalized and st.session_state.logged_in:
        st.markdown("### 🔮 Your Personalized Collaborative Feed (SVD Model)")
        st.write("Recommended specifically for you based on SVD matrix factorization predictions:")
        
        collab_recs, err_collab = api_get("/recommend/collaborative", {
            "user_id": st.session_state.user_id,
            "top_n": 18
        })
        if not err_collab and collab_recs:
            render_movie_grid(collab_recs, cols=grid_cols, key_prefix="collab_feed")
        else:
            st.info("We are preparing your personalized feed. Start rating movies to update your tastes!")
            
    # Regular browse categories with language/genre discover filters
    else:
        lang_code = None
        if lang_filter != "All":
            lang_code = lang_filter.split("(")[-1].replace(")", "").strip()
            
        genre_id = GENRE_MAP[genre_filter]
        
        if lang_code or genre_id:
            st.markdown(f"### 🌐 Discovering — Language: {lang_filter} | Genre: {genre_filter}")
            discover_cards, err_disc = api_get("/home/discover", {
                "language": lang_code,
                "genre": genre_id,
                "limit": 24
            })
            if not err_disc and discover_cards:
                render_movie_grid(discover_cards, cols=grid_cols, key_prefix="discover_feed")
            else:
                st.write("Could not discover movies matching filters.")
        else:
            # Standard category selector feed
            category_mapping = {
                "Trending Daily": "trending",
                "Popular": "popular",
                "Top Rated": "top_rated",
                "Now Playing": "now_playing",
                "Upcoming": "upcoming"
            }
            sub_category = st.tabs(list(category_mapping.keys()))
            
            for t_idx, tab_name in enumerate(category_mapping.keys()):
                with sub_category[t_idx]:
                    cat_code = category_mapping[tab_name]
                    home_cards, err_hc = api_get("/home", {"category": cat_code, "limit": 24})
                    if not err_hc and home_cards:
                        render_movie_grid(home_cards, cols=grid_cols, key_prefix=f"feed_{cat_code}")
                    else:
                        st.error("Failed to load feed.")

# ==========================================================
# VIEW: AI CHATBOT
# ==========================================================
elif st.session_state.page == "🤖 AI Chatbot Assistant":
    st.markdown("<h1 style='color:#38bdf8;'>🤖 Conversational AI Chatbot</h1>", unsafe_allow_html=True)
    st.write("Chat with our LLM Movie Recommender to find movies based on storylines, moods, directors, actors, or themes.")
    
    # Context attachment
    movie_context_info = None
    if st.session_state.selected_tmdb_id:
        details_c, _ = api_get(f"/movie/id/{st.session_state.selected_tmdb_id}")
        if details_c:
            movie_context_info = f"Title: {details_c['title']}, Overview: {details_c['overview']}"
            st.info(f"📎 Current Movie Context Attached: **{details_c['title']}**")
    
    # Add a Clear Chat button
    col_chat_title, col_chat_clear = st.columns([5, 1.2])
    with col_chat_clear:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
            
    # Display chat messages from history
    for chat in st.session_state.chat_history:
        avatar = "👤" if chat["role"] == "user" else "🍿"
        with st.chat_message(chat["role"], avatar=avatar):
            st.markdown(chat["content"])
            
    # React to user input
    if chat_msg := st.chat_input("Ask VibeRec Assistant for movie recommendations..."):
        # Display user message in chat container
        with st.chat_message("user", avatar="👤"):
            st.markdown(chat_msg)
            
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": chat_msg})
        
        # Display assistant response with loading spinner
        with st.chat_message("assistant", avatar="🍿"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                res, _ = api_post("/chatbot", {
                    "message": chat_msg,
                    "history": st.session_state.chat_history[:-1],
                    "current_movie_context": movie_context_info
                })
                
                if res and "response" in res:
                    response_text = res["response"]
                else:
                    response_text = "Sorry, I am offline right now."
                    
            message_placeholder.markdown(response_text)
            
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.rerun()

# ==========================================================
# VIEW: MOOD BASED FINDER
# ==========================================================
elif st.session_state.page == "🎭 Mood-based Finder":
    st.markdown("<h1 style='color:#38bdf8;'>🎭 Find Movies by Mood</h1>", unsafe_allow_html=True)
    st.write("How are you feeling right now? Tap a mood to get matching movie recommendations dynamically:")
    
    moods = {
        "Happy 😃": "happy",
        "Sad 🥺": "sad",
        "Excited 🤩": "excited",
        "Romantic 🥰": "romantic",
        "Thrilling 😰": "thrilling",
        "Relaxed 😌": "relaxed",
        "Nostalgic 🕰️": "nostalgic",
        "Scared 👻": "scared"
    }
    
    # Render mood buttons
    m_cols = st.columns(4)
    selected_mood_val = None
    for idx, (label, code) in enumerate(moods.items()):
        c_idx = idx % 4
        with m_cols[c_idx]:
            if st.button(label, use_container_width=True, key=f"mood_btn_{code}"):
                selected_mood_val = code
                st.session_state.selected_mood = code
                
    if "selected_mood" in st.session_state:
        m_code = st.session_state.selected_mood
        st.divider()
        st.markdown(f"### Vibe Feed — Feeling {m_code.capitalize()}")
        
        # Call explain/commentary helper
        explain_commentary, _ = api_get("/explain", {
            "title": f"{m_code} mood selections",
            "rec_type": "mood",
            "details": f"User mood is {m_code}"
        })
        if explain_commentary:
            st.info(explain_commentary.get("explanation", ""))
            
        # Fetch mood recommendations
        mood_cards, err_mood = api_get("/recommend/mood", {"mood": m_code, "limit": 18})
        if not err_mood and mood_cards:
            render_movie_grid(mood_cards, cols=grid_cols, key_prefix="mood_results")
        else:
            st.error("Could not fetch mood movies.")

# ==========================================================
# VIEW: MY WISHLIST
# ==========================================================
elif st.session_state.page == "❤️ My Wishlist":
    st.markdown("<h1 style='color:#38bdf8;'>❤️ My Wishlist</h1>", unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("Please login in the sidebar to view your Wishlist.")
    else:
        wishlist_items, err_w = api_get(f"/user/{st.session_state.user_id}/wishlist")
        if not err_w and wishlist_items:
            cards = []
            for item in wishlist_items:
                cards.append({
                    "tmdb_id": item["tmdb_id"],
                    "title": item["title"],
                    "poster_url": item["poster_url"],
                    "vote_average": 0.0 # simple card shape
                })
            render_movie_grid(cards, cols=grid_cols, key_prefix="wishlist_grid")
        else:
            st.info("Your wishlist is empty. Add movies you want to watch from the details page!")

# ==========================================================
# VIEW: WATCH HISTORY
# ==========================================================
elif st.session_state.page == "📜 Watch History":
    st.markdown("<h1 style='color:#38bdf8;'>📜 Watch History</h1>", unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("Please login in the sidebar to view your Watch History.")
    else:
        history_items, err_h = api_get(f"/user/{st.session_state.user_id}/history")
        if not err_h and history_items:
            cards = []
            for item in history_items:
                cards.append({
                    "tmdb_id": item["tmdb_id"],
                    "title": item["title"],
                    "poster_url": item["poster_url"],
                    "vote_average": 0.0
                })
            st.caption("Here are the movies you recently viewed:")
            render_movie_grid(cards, cols=grid_cols, key_prefix="history_grid")
        else:
            st.info("No watch history yet. Start browsing movie details to build your history!")

# ==========================================================
# VIEW: MY RATINGS
# ==========================================================
elif st.session_state.page == "⭐ My Ratings & Reviews":
    st.markdown("<h1 style='color:#38bdf8;'>⭐ My Ratings & Reviews</h1>", unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        st.warning("Please login in the sidebar to view your ratings.")
    else:
        ratings_items, err_r = api_get(f"/user/{st.session_state.user_id}/ratings")
        if not err_r and ratings_items:
            st.write("Here are the movies you rated and reviewed:")
            for item in ratings_items:
                # Fetch details for the name
                m_details, _ = api_get(f"/movie/id/{item['tmdb_id']}")
                m_title = m_details.get("title", f"Movie #{item['tmdb_id']}") if m_details else f"Movie #{item['tmdb_id']}"
                
                st.markdown("<div class='info-panel'>", unsafe_allow_html=True)
                r_col1, r_col2 = st.columns([1, 5])
                with r_col1:
                    if m_details and m_details.get("poster_url"):
                        st.image(m_details["poster_url"], width=100)
                with r_col2:
                    st.markdown(f"### {m_title}")
                    st.markdown(f"**Rating:** {'⭐' * int(float(item['rating']))} ({item['rating']}/5.0)")
                    st.markdown(f"**Review:** {item['review_text'] or '*No review text written.*'}")
                    st.markdown(f"<span style='color:#94a3b8; font-size:0.8rem;'>Rated on: {item['created_at']}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("You haven't rated any movies yet. Open a movie details page to submit a rating!")

# ==========================================================
# VIEW: SYSTEM EVALUATION DASHBOARD
# ==========================================================
elif st.session_state.page == "📊 System Evaluation":
    st.markdown("<h1 style='color:#38bdf8;'>📊 Recommendation System Evaluation</h1>", unsafe_allow_html=True)
    st.write("Evaluation metrics computed on the MovieLens-small test set (80-20 rating split) for the collaborative SVD recommendation engine:")
    
    metrics_data, err_m = api_get("/evaluate/metrics")
    
    if not err_m and metrics_data:
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>RMSE</div><div class='metric-value'>{metrics_data['rmse']:.3f}</div><div style='font-size:0.75rem; color:#94a3b8; margin-top:5px;'>Lower is better</div></div>",
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>Precision@10</div><div class='metric-value'>{metrics_data['precision_k']:.3f}</div><div style='font-size:0.75rem; color:#94a3b8; margin-top:5px;'>Higher is better</div></div>",
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>Recall@10</div><div class='metric-value'>{metrics_data['recall_k']:.3f}</div><div style='font-size:0.75rem; color:#94a3b8; margin-top:5px;'>Higher is better</div></div>",
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>NDCG@10</div><div class='metric-value'>{metrics_data['ndcg_k']:.3f}</div><div style='font-size:0.75rem; color:#94a3b8; margin-top:5px;'>Higher is better</div></div>",
                unsafe_allow_html=True
            )
        with m5:
            st.markdown(
                f"<div class='metric-box'><div class='metric-title'>MAP</div><div class='metric-value'>{metrics_data['map']:.3f}</div><div style='font-size:0.75rem; color:#94a3b8; margin-top:5px;'>Higher is better</div></div>",
                unsafe_allow_html=True
            )
            
        st.divider()
        
        # Display chart
        st.subheader("📈 Ranking Quality Metrics Visualization")
        chart_data = pd.DataFrame({
            'Metric': ['Precision@10', 'Recall@10', 'NDCG@10', 'MAP'],
            'Score Value': [metrics_data['precision_k'], metrics_data['recall_k'], metrics_data['ndcg_k'], metrics_data['map']]
        }).set_index('Metric')
        st.bar_chart(chart_data, color='#38bdf8')
        
        st.info(f"ℹ️ Computed metrics dynamically on a subset of the **{metrics_data['evaluated_records']}** test ratings from MovieLens.")
    else:
        st.error(f"Could not load evaluation metrics: {err_m}")
