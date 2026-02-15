import streamlit as st 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movies.csv")
movies['genres_clean'] = movies['genres'].str.replace('|', " ")

tfidf= TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(movies['genres_clean'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, top_n=5):
    movie_index = movies[movies['title']==title].index[0]
    sim_scores=list(enumerate(cosine_sim[movie_index]))
    sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_sim_scores = sim_scores[1: top_n+1]
    
    recommended_indices=[]
    for i in top_sim_scores:
        recommended_indices.append(i[0])
    recommended_titles = movies['title'].iloc[recommended_indices].tolist()
    return recommended_titles

#streamlit app
st.title("Movie Recommender")
selected_movie = st.selectbox("Choose a movie:", movies['title'].values)
top_n = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Recommend"):
    recommendations = recommend(selected_movie, top_n)
    st.write("Top recommendations:")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")