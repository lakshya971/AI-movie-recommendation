import os
import sqlite3
import hashlib
from datetime import datetime

DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "movie_rec.db")

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create ratings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        tmdb_id INTEGER NOT NULL,
        rating REAL NOT NULL,
        review_text TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        UNIQUE(user_id, tmdb_id)
    )
    """)
    
    # Create watch_history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS watch_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        tmdb_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        poster_url TEXT,
        watched_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        UNIQUE(user_id, tmdb_id)
    )
    """)
    
    # Create wishlist table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS wishlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        tmdb_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        poster_url TEXT,
        added_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        UNIQUE(user_id, tmdb_id)
    )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def register_user(username: str, password: str) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, pwd_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Username already exists")

def authenticate_user(username: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    pwd_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
        (username, pwd_hash)
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"id": user["id"], "username": user["username"]}
    return None

def add_rating(user_id: int, tmdb_id: int, rating: float, review_text: str = ""):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO ratings (user_id, tmdb_id, rating, review_text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, tmdb_id) DO UPDATE SET
            rating = excluded.rating,
            review_text = excluded.review_text,
            created_at = CURRENT_TIMESTAMP
        """,
        (user_id, tmdb_id, rating, review_text)
    )
    conn.commit()
    conn.close()

def get_user_ratings(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT tmdb_id, rating, review_text, created_at FROM ratings WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_to_history(user_id: int, tmdb_id: int, title: str, poster_url: str = ""):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO watch_history (user_id, tmdb_id, title, poster_url)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, tmdb_id) DO UPDATE SET
            watched_at = CURRENT_TIMESTAMP
        """,
        (user_id, tmdb_id, title, poster_url)
    )
    conn.commit()
    conn.close()

def get_history(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT tmdb_id, title, poster_url, watched_at FROM watch_history WHERE user_id = ? ORDER BY watched_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def toggle_wishlist(user_id: int, tmdb_id: int, title: str, poster_url: str = "") -> bool:
    """Returns True if added, False if removed"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM wishlist WHERE user_id = ? AND tmdb_id = ?",
        (user_id, tmdb_id)
    )
    exists = cursor.fetchone()
    
    if exists:
        cursor.execute(
            "DELETE FROM wishlist WHERE user_id = ? AND tmdb_id = ?",
            (user_id, tmdb_id)
        )
        added = False
    else:
        cursor.execute(
            "INSERT INTO wishlist (user_id, tmdb_id, title, poster_url) VALUES (?, ?, ?, ?)",
            (user_id, tmdb_id, title, poster_url)
        )
        added = True
        
    conn.commit()
    conn.close()
    return added

def get_wishlist(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT tmdb_id, title, poster_url, added_at FROM wishlist WHERE user_id = ? ORDER BY added_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def check_wishlist_status(user_id: int, tmdb_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM wishlist WHERE user_id = ? AND tmdb_id = ?",
        (user_id, tmdb_id)
    )
    exists = cursor.fetchone()
    conn.close()
    return exists is not None
