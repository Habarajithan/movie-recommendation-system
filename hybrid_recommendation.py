import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

print("Running Hybrid Recommendation System...")

# ==============================
# 1. Load Data
# ==============================
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# Ensure required columns exist
if 'overview' not in movies.columns:
    movies['overview'] = ""

if 'year' not in movies.columns:
    movies['year'] = 2000

# ==============================
# 2. Content-Based Model
# ==============================
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'].str.split('|'))

# Use overview + genres if overview exists
if movies['overview'].str.strip().replace("", np.nan).isna().all():
    print("Overview empty → using genres only")
    content_matrix = genre_matrix
else:
    tfidf = TfidfVectorizer(stop_words='english')
    overview_matrix = tfidf.fit_transform(movies['overview'])
    content_matrix = np.hstack([overview_matrix.toarray(), genre_matrix])

cos_sim = cosine_similarity(content_matrix)
print("Content-based model ready ✅")

# ==============================
# 3. Collaborative Filtering (SVD)
# ==============================
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']], reader
)

param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [20, 30],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(data)
best_params = gs.best_params['rmse']

svd = SVD(**best_params)
svd.fit(data.build_full_trainset())
print("Collaborative model ready ✅")

# ==============================
# 4. Popularity & Recency
# ==============================
popularity = ratings.groupby('movieId')['rating'].count().to_dict()
max_pop = max(popularity.values())

movies['pop_score'] = movies['movieId'].apply(
    lambda x: popularity.get(x, 0) / max_pop
)

movies['recency_score'] = (
    (movies['year'] - movies['year'].min()) /
    (movies['year'].max() - movies['year'].min())
)

movieId_to_index = {
    mid: idx for idx, mid in enumerate(movies['movieId'])
}

# ==============================
# 5. Hybrid Recommendation
# ==============================
def recommend_hybrid(user_id, top_n=5,
                     w_cf=0.5, w_content=0.3, w_pop=0.1, w_rec=0.1):

    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    candidates = movies[~movies['movieId'].isin(rated_movies)]

    results = []

    for _, row in candidates.iterrows():
        mid = row['movieId']
        idx = movieId_to_index[mid]

        # CF score
        cf_score = svd.predict(user_id, mid).est

        # Content similarity score
        sim_scores = [
            cos_sim[idx, movieId_to_index[rm]]
            for rm in rated_movies if rm in movieId_to_index
        ]
        content_score = max(sim_scores) if sim_scores else 0

        final_score = (
            w_cf * cf_score +
            w_content * content_score +
            w_pop * row['pop_score'] +
            w_rec * row['recency_score']
        )

        results.append((row['title'], final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in results[:top_n]]

# ==============================
# 6. Cold-Start: New User (Quiz)
# ==============================
def recommend_cold_start_user(preferred_genres, top_n=5):
    genre_indices = [
        mlb.classes_.tolist().index(g)
        for g in preferred_genres if g in mlb.classes_
    ]

    scores = []
    for idx, row in movies.iterrows():
        genre_score = (
            genre_matrix[idx, genre_indices].sum()
            if genre_indices else 0
        )

        final_score = (
            0.6 * genre_score +
            0.2 * row['pop_score'] +
            0.2 * row['recency_score']
        )

        scores.append((row['title'], final_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scores[:top_n]]

# ==============================
# 7. Cold-Start: New Item
# ==============================
def recommend_for_new_item(movie_id, top_n=5):
    if movie_id not in movieId_to_index:
        return []

    idx = movieId_to_index[movie_id]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, _ in sim_scores[1:top_n+1]]
    return movies.iloc[top_indices]['title'].tolist()

# ==============================
# 8. Test Runs
# ==============================
print("\nHybrid Recommendations (Existing User):")
hybrid_movies = recommend_hybrid(user_id=1, top_n=5)
for i, m in enumerate(hybrid_movies, 1):
    print(f"{i}. {m}")

print("\nCold-Start (New User):")
cold_user_movies = recommend_cold_start_user(
    ['Action', 'Sci-Fi'], top_n=5
)
for i, m in enumerate(cold_user_movies, 1):
    print(f"{i}. {m}")

print("\nCold-Start (New Item):")
cold_item_movies = recommend_for_new_item(
    movies['movieId'].iloc[0], top_n=5
)
for i, m in enumerate(cold_item_movies, 1):
    print(f"{i}. {m}")

print("\nHybrid recommendation system complete ✅")