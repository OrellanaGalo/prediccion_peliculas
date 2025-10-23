import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Iniciando entrenamiento del modelo...")

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

top_movies = ratings['movieId'].value_counts().head(1000).index
top_users = ratings['userId'].value_counts().head(1000).index
ratings_small = ratings[ratings['movieId'].isin(top_movies) & ratings['userId'].isin(top_users)]

print("Datos cargados y filtrados")

user_item_matrix = ratings_small.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
print("Matriz usuario-pelicula creada")

cosine_sim = cosine_similarity(user_item_matrix.T)
cosine_sim_df = pd.DataFrame(
    cosine_sim,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)
print("Matriz de similitud del coseno calculada.")

with open('modelo_similitud.pkl', 'wb') as f:
    pickle.dump(cosine_sim_df, f)

movies.to_pickle('movies_df.pkl')

print("-" * 30)
print("Modelo entrenado y guardado exitosamente")
print("Archivos generados: 'modelo_similitud.pkl' y 'movies_df.pkl'")