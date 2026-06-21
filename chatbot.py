import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_groq_client():
    if not GROQ_API_KEY:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error creating Groq client: {e}")
        return None

_df_cache = None

def get_local_movies_df():
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    import pandas as pd
    import pickle
    df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "df.pkl")
    if os.path.exists(df_path):
        try:
            with open(df_path, "rb") as f:
                _df_cache = pickle.load(f)
            return _df_cache
        except Exception:
            pass
    return None

def local_fallback_chat(message: str, current_movie_context: str = None) -> str:
    df = get_local_movies_df()
    msg_lower = message.lower()
    
    # Identify genre keywords
    keywords = {
        "romance": ["romance", "romantic", "love"],
        "action": ["action", "fight", "explosion", "thrill", "hero"],
        "comedy": ["comedy", "funny", "laugh", "hilarious", "humor"],
        "horror": ["horror", "scary", "ghost", "spooky", "vampire", "zombie"],
        "sci-fi": ["sci-fi", "science fiction", "space", "alien", "future", "robot"],
        "drama": ["drama", "sad", "emotional", "tragic", "relationships"],
        "animation": ["animation", "animated", "cartoon", "disney", "pixar"],
        "thriller": ["thriller", "mystery", "suspense", "crime"]
    }
    
    found_genre = None
    for genre, terms in keywords.items():
        if any(term in msg_lower for term in terms):
            found_genre = genre
            break
            
    matched_movies = []
    if df is not None and not df.empty:
        # Search by genre keyword if found
        if found_genre:
            mask = df['genres'].astype(str).str.lower().str.contains(found_genre)
            matches = df[mask].head(5)
            for _, row in matches.iterrows():
                matched_movies.append(f"🍿 **{row['title']}** - {row['overview'][:150]}...")
        
        # If no genre matched or no matches, search by keywords in message
        if not matched_movies:
            words = [w for w in msg_lower.split() if len(w) > 3]
            if words:
                # Build compound query
                mask = df['title'].astype(str).str.lower().str.contains(words[0]) | \
                       df['overview'].astype(str).str.lower().str.contains(words[0])
                for w in words[1:]:
                    mask = mask | df['title'].astype(str).str.lower().str.contains(w) | \
                           df['overview'].astype(str).str.lower().str.contains(w)
                matches = df[mask].head(5)
                for _, row in matches.iterrows():
                    matched_movies.append(f"🍿 **{row['title']}** - {row['overview'][:150]}...")
                    
    # Generate response
    response = "*(Offline Recommender Mode - Groq API Unavailable)*\n\n"
    if matched_movies:
        genre_str = f" for **{found_genre}**" if found_genre else ""
        response += f"I'm operating in offline mode right now, but here are some movies matching your request{genre_str} from our database:\n\n"
        response += "\n\n".join(matched_movies)
    else:
        response += "I'm currently unable to connect to the AI engine, and couldn't find matching movies for your specific keywords. Try searching for genres like *Action*, *Comedy*, *Romance*, *Sci-Fi*, or *Horror*."
        
    return response

def chat_with_llm(message: str, history: list, current_movie_context: str = None) -> str:
    """
    Conversational AI Chatbot with local database fallback.
    """
    client = get_groq_client()
    if not client:
        return local_fallback_chat(message, current_movie_context)
        
    system_prompt = (
        "You are an expert AI Movie Recommendation Chatbot. Your goal is to help users find movies "
        "they will love based on their preferences, mood, actors, directors, or specific plot elements.\n"
        "Be friendly, conversational, and concise. Always suggest specific movie titles, and explain briefly why they fit the user's request."
    )
    
    if current_movie_context:
        system_prompt += f"\nNote: The user is currently viewing details for the movie: {current_movie_context}. Feel free to reference it if relevant."
        
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    for chat in history[-6:]: # Keep last 6 exchanges to prevent context bloat
        messages.append({
            "role": chat["role"],
            "content": chat["content"]
        })
        
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API error, falling back to local search: {e}")
        return local_fallback_chat(message, current_movie_context)

# Static mapping for moods as reliable fallback / baseline
MOOD_GENRE_MAPPING = {
    "happy": [35, 10751, 10402],    # Comedy, Family, Music
    "sad": [18, 35, 10749],         # Drama, Comedy (to lift mood), Romance
    "excited": [28, 12, 878],       # Action, Adventure, Sci-Fi
    "romantic": [10749, 18],        # Romance, Drama
    "thrilling": [53, 9648, 80],    # Thriller, Mystery, Crime
    "relaxed": [99, 14, 16],        # Documentary, Fantasy, Animation
    "nostalgic": [36, 37, 10751],    # History, Western, Family
    "scared": [27, 53]              # Horror, Thriller
}

def generate_mood_commentary(mood: str) -> str:
    """Generates a short personalized greeting for a user's mood."""
    client = get_groq_client()
    if not client:
        return f"Here are some great movies to match your {mood} mood!"
        
    prompt = f"Write a warm, 1-2 sentence greeting for a user who selected the mood '{mood}' in a movie recommender app, explaining what kind of vibe we are going for."
    
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=100
        )
        return completion.choices[0].message.content
    except Exception:
        return f"Here are some great movies to match your {mood} mood!"

def explain_recommendation(movie_title: str, rec_type: str, details: str = "") -> str:
    """
    Explainable AI: Generates a friendly 2-sentence explanation of why a movie is recommended.
    """
    client = get_groq_client()
    if not client:
        # Static explanation generator fallback
        if rec_type.lower() == "collaborative":
            return f"We recommended '{movie_title}' because other users with similar viewing and rating habits enjoyed it."
        elif rec_type.lower() == "tfidf":
            return f"We recommended '{movie_title}' because it has plot themes and descriptions highly similar to the movie you selected."
        elif rec_type.lower() == "sentence":
            return f"This movie shares deep narrative connections, moods, and semantic plot topics with your selection."
        elif rec_type.lower() == "knowledge":
            return f"This fits your personal preferences, matching your highly rated genres and directors."
        else:
            return f"Based on your preferences, this is a highly recommended watch."
            
    prompt = (
        f"In a movie recommender app, explain to the user in a warm, concise, and friendly way "
        f"(maximum 2 sentences) why the movie '{movie_title}' is recommended to them as a "
        f"'{rec_type}' recommendation. "
    )
    if details:
        prompt += f"Context/Details to use: {details}."
        
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=120
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Recommended based on its strong similarity and match with your profile."
