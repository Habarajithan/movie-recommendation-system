import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV, train_test_split

# 1. Load ratings and movies data
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# 2. Prepare data for Surprise
# MovieLens rating scale is 0.5 to 5
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(
    ratings[['userId', 'movieId', 'rating']],
    reader
)

# 3. Grid search for best SVD parameters
param_grid = {
    'n_factors': [50, 100],     # latent factors
    'n_epochs': [20, 30],       # training epochs
    'lr_all': [0.002, 0.005],   # learning rate
    'reg_all': [0.02, 0.1]      # regularization
}

gs = GridSearchCV(
    SVD,
    param_grid,
    measures=['rmse'],
    cv=3
)
gs.fit(data)

best_params = gs.best_params['rmse']
print("Best SVD parameters:", best_params)

# 4. Model Evaluation using Train-Test Split (RMSE)
trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

svd_temp = SVD(**best_params)
svd_temp.fit(trainset)

predictions = svd_temp.test(testset)
rmse = accuracy.rmse(predictions)
print("Test RMSE:", rmse)

# 5. Train final SVD model on full dataset
full_trainset = data.build_full_trainset()
svd = SVD(**best_params)
svd.fit(full_trainset)

# Get movies already rated by the user
def get_unrated_movies(user_id, ratings_df, movies_df):
    rated_movie_ids = ratings_df[
        ratings_df['userId'] == user_id
    ]['movieId'].tolist()

    unrated_movies = movies_df[
        ~movies_df['movieId'].isin(rated_movie_ids)
    ]
    return unrated_movies

# 6. Recommendation function
def recommend_svd(user_id, ratings_df, movies_df, top_n=5):
    """
    Recommend top_n unseen movies for a given user
    """
    # Cold-start handling
    if user_id not in ratings_df['userId'].values:
        return ["New user – no recommendations available"]

    unrated_movies = get_unrated_movies(
        user_id,
        ratings_df,
        movies_df
    )

    predictions = []
    for _, row in unrated_movies.iterrows():
        pred = svd.predict(user_id, row['movieId'])
        predictions.append((row['title'], pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return [movie for movie, _ in predictions[:top_n]]

# 7. Test run
user_id = 1
top_n = 5

top_movies = recommend_svd(
    user_id,
    ratings,
    movies,
    top_n
)

print(f"\nTop {top_n} recommended movies for User {user_id}:")
for i, movie in enumerate(top_movies, 1):
    print(f"{i}. {movie}")