import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

top_movies = ratings['movieId'].value_counts().head(1000).index
top_users = ratings['userId'].value_counts().head(1000).index
ratings_small = ratings[ratings['movieId'].isin(top_movies) & ratings['userId'].isin(top_users)]

user_item_matrix = ratings_small.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

cosine_sim = cosine_similarity(user_item_matrix.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recomendar_peliculas(movie_title, movies_df, similarity_df, top_n=5):
    movie_id = movies_df[movies_df['title'] == movie_title]['movieId'].values
    if len(movie_id) == 0:
        return f"La película '{movie_title}' no se encontró en el dataset."
    
    movie_id = movie_id[0]
    similar_movies = similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_movies.drop(movie_id)

    top_movies = similar_movies.head(top_n).index
    top_movie_titles = movies_df[movies_df['movieId'].isin(top_movies)]['title'].values
    return list(top_movie_titles)

pelicula = "Star Wars: Episode IV - A New Hope (1977)"
recomendadas = recomendar_peliculas(pelicula, movies, cosine_sim_df)
print(f"Películas recomendadas para '{pelicula}':\n", recomendadas)
