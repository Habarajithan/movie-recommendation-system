# ==============================
# evaluation.py
# ==============================

import pandas as pd
import numpy as np
from collections import defaultdict

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# LOAD DATA
# ==============================

ratings = pd.read_csv("data/ratings.csv")   # userId, movieId, rating
movies = pd.read_csv("data/movies.csv")     # movieId, title, overview

print("Data loaded successfully")


# ==============================
# CONTENT-BASED MODEL
# ==============================

# Create text feature using title + genres
tfidf = TfidfVectorizer(stop_words="english")
movies["text"] = movies["title"] + " " + movies["genres"]

tfidf_matrix = tfidf.fit_transform(movies["text"].fillna(""))

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_indices = pd.Series(movies.index, index=movies["movieId"])


def content_based_recommend(user_id, k=10):
    user_movies = ratings[ratings["userId"] == user_id]["movieId"]
    scores = np.zeros(len(movies))

    for movie in user_movies:
        if movie in movie_indices:
            idx = movie_indices[movie]
            scores += cosine_sim[idx]

    top_indices = scores.argsort()[-k:][::-1]
    return movies.iloc[top_indices]["movieId"].values


# ==============================
# COLLABORATIVE FILTERING (SVD)
# ==============================

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

trainset, testset = train_test_split(data, test_size=0.2)

svd = SVD()
svd.fit(trainset)


def collaborative_recommend(user_id, k=10):
    movie_ids = ratings["movieId"].unique()
    predictions = []

    for movie in movie_ids:
        pred = svd.predict(user_id, movie).est
        predictions.append((movie, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in predictions[:k]]


# ==============================
# HYBRID MODEL
# ==============================

def hybrid_recommend(user_id, k=10, alpha=0.5):
    cb_recs = content_based_recommend(user_id, k * 2)
    cf_recs = collaborative_recommend(user_id, k * 2)

    scores = defaultdict(float)

    for rank, movie in enumerate(cb_recs):
        scores[movie] += alpha * (1 / (rank + 1))

    for rank, movie in enumerate(cf_recs):
        scores[movie] += (1 - alpha) * (1 / (rank + 1))

    ranked_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in ranked_movies[:k]]


# ==============================
# EVALUATION METRICS
# ==============================

def precision_recall_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    relevant = set(relevant)

    tp = len(set(recommended) & relevant)

    precision = tp / k if k else 0
    recall = tp / len(relevant) if relevant else 0

    return precision, recall


def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0

    for i, movie in enumerate(recommended[:k]):
        if movie in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_dcg = sum(
        1 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )

    return dcg / ideal_dcg if ideal_dcg > 0 else 0


def average_precision(recommended, relevant):
    score = 0.0
    hits = 0

    for i, movie in enumerate(recommended):
        if movie in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / len(relevant) if len(relevant) > 0 else 0


# ==============================
# MODEL EVALUATION FUNCTION
# ==============================

def evaluate_model(recommender_function, k=10):
    precisions = []
    recalls = []
    ndcgs = []
    aps = []

    users = ratings["userId"].unique()

    for user in users:
        relevant_movies = ratings[
            (ratings["userId"] == user) &
            (ratings["rating"] >= 4)
        ]["movieId"].values

        if len(relevant_movies) == 0:
            continue

        recommended_movies = recommender_function(user, k)

        p, r = precision_recall_at_k(
            recommended_movies, relevant_movies, k
        )

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg_at_k(
            recommended_movies, relevant_movies, k
        ))
        aps.append(average_precision(
            recommended_movies, relevant_movies
        ))

    return (
        np.mean(precisions),
        np.mean(recalls),
        np.mean(ndcgs),
        np.mean(aps)
    )


# ==============================
# RUN EVALUATION
# ==============================

K = 10

print("\nRunning evaluation... Please wait\n")

content_results = evaluate_model(content_based_recommend, K)
collab_results = evaluate_model(collaborative_recommend, K)
hybrid_results = evaluate_model(hybrid_recommend, K)

print("====== Evaluation Results (k=10) ======\n")

print("Content-Based Model")
print("Precision@10:", content_results[0])
print("Recall@10   :", content_results[1])
print("nDCG@10     :", content_results[2])
print("MAP         :", content_results[3])

print("\nCollaborative Model (SVD)")
print("Precision@10:", collab_results[0])
print("Recall@10   :", collab_results[1])
print("nDCG@10     :", collab_results[2])
print("MAP         :", collab_results[3])

print("\nHybrid Model")
print("Precision@10:", hybrid_results[0])
print("Recall@10   :", hybrid_results[1])
print("nDCG@10     :", hybrid_results[2])
print("MAP         :", hybrid_results[3])