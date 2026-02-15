import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

#pandas
movies = pd.read_csv("data/movies.csv")
tags = pd.read_csv("data/tags.csv")

#cleaning genres
movies["genres_clean"]=movies["genres"].str.replace("|", " ", regex=False)

#combining all tags for each movies
tags_grouped=(tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index())

#merging tags with movies
movies = movies.merge(tags_grouped, on="movieId", how="left")
movies["tag"]=movies["tag"].fillna("")

#TD-IDF on genres
genre_tfidf = TfidfVectorizer(stop_words="english")
genre_tfidf_matrix=genre_tfidf.fit_transform(movies["genres_clean"])

#TF-IDF on tags
tag_tfidf = TfidfVectorizer(stop_words="english")
tag_tfifdf_matrix=tag_tfidf.fit_transform(movies['tag'])

#combining
combined_matrix = hstack([genre_tfidf_matrix, tag_tfifdf_matrix])

#cosine similarity
cosine_sim = cosine_similarity(combined_matrix, combined_matrix)


#recommendation function
def recommend(title, top_n=5):
    if title not in movies["title"].values:
        return ["Movie not found"]

    idx=movies[movies['title']==title].index[0]#we're getting index of each movie
    sim_scores=list(enumerate(cosine_sim[idx]))#to get pairwise similarity scores
    sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)#sort movies by similarity scores(giving highest first)
    top_sim_scores = sim_scores[1: top_n+1]#get top_n
    
    recommended_indices=[]
    for i in top_sim_scores:
        recommended_indices.append(i[0])
    recommended_titles = movies['title'].iloc[recommended_indices].tolist()
    return recommended_titles

movie_name = "Toy Story (1995)"
top_n = 5
result = recommend(movie_name, 5)
print(f"\nMovies similar to '{movie_name}':\n")
count = 1
for movie in result:
    print(f"{count}. {movie}")
    count+=1
