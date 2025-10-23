import pandas as pd
import pickle

print("Cargando modelo de recomendación...")
try:
    with open('modelo_similitud.pkl', 'rb') as f:
        cosine_sim_df = pickle.load(f)
    
    movies = pd.read_pickle('movies_df.pkl')
    print("¡Modelo cargado exitosamente!")
except FileNotFoundError:
    print("Error: No se encontraron los archivos del modelo.")
    print("Por favor, ejecuta primero 'entrenar_modelo.py' para generar el modelo.")
    exit()

def obtener_recomendaciones_ids(movie_id, similarity_df, top_n=10):
    """Obtiene una lista de IDs de películas recomendadas."""
    if movie_id not in similarity_df.index:
        return []
    
    similar_movies = similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_movies.drop(movie_id)
    top_movies_ids = similar_movies.head(top_n).index
    return list(top_movies_ids)

def recomendar_peliculas_por_titulo(movie_title, movies_df, similarity_df, top_n=5):
    """Función final que busca por título y devuelve una lista de títulos."""
    movie_id_series = movies_df[movies_df['title'] == movie_title]['movieId']
    
    if movie_id_series.empty:
        return f"La película '{movie_title}' no se encontró en el dataset."
    
    movie_id = movie_id_series.values[0]
    
    recommended_ids = obtener_recomendaciones_ids(movie_id, similarity_df, top_n)
    
    if not recommended_ids:
        return "No se pueden generar recomendaciones para esta película (puede que no estuviera en los datos de entrenamiento)."
    
    recommended_titles = movies_df[movies_df['movieId'].isin(recommended_ids)]['title'].tolist()
    return recommended_titles

if __name__ == "__main__":
    pelicula_ejemplo = "Star Wars: Episode IV - A New Hope (1977)"
    
    print("-" * 30)
    print(f"Buscando recomendaciones para: '{pelicula_ejemplo}'")
    
    recomendadas = recomendar_peliculas_por_titulo(pelicula_ejemplo, movies, cosine_sim_df)

    if isinstance(recomendadas, list):
        print("Películas recomendadas:")
        for i, title in enumerate(recomendadas):
            print(f"{i+1}. {title}")
    else:
        print(recomendadas)