from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Movie Recommendation API")

# -------------------------------
# Load Data
# -------------------------------
movies = pd.read_csv("data/movies.csv")

movies["genres"] = movies["genres"].fillna("")

# TF-IDF on genres (tags)
tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------------
# Test Users
# -------------------------------
TEST_USERS = {
    "user1": {"id": 1, "name": "Test User 1"},
    "user2": {"id": 2, "name": "Test User 2"}
}

# -------------------------------
# Helper Functions
# -------------------------------
def get_top_n_movies(n=10):
    return movies.sample(n)[["movieId", "title"]]

def get_similar_movies(movie_id, top_n=5):
    idx = movies.index[movies["movieId"] == movie_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, score in sim_scores:
        results.append({
            "movieId": int(movies.iloc[i]["movieId"]),
            "title": movies.iloc[i]["title"],
            "similarity": round(score, 3)
        })
    return results

def explain_recommendation(movie_id):
    movie = movies[movies["movieId"] == movie_id].iloc[0]
    genres = set(movie["genres"].split("|"))

    return {
        "title": movie["title"],
        "genres": list(genres),
        "why_recommended": "Recommended because it matches genres you like: " + ", ".join(genres)
    }

# -------------------------------
# API Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running 🚀"}

@app.get("/login/{username}")
def login(username: str):
    if username not in TEST_USERS:
        raise HTTPException(status_code=401, detail="Invalid test user")
    return {"message": "Login successful", "user": TEST_USERS[username]}

@app.get("/recommendations/top-n")
def top_n(n: int = 10):
    return get_top_n_movies(n).to_dict(orient="records")

@app.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: int, n: int = 5):
    return get_similar_movies(movie_id, n)

@app.get("/movies/{movie_id}/why")
def why_recommended(movie_id: int):
    return explain_recommendation(movie_id)