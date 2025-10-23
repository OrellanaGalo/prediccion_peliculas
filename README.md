# Prediccion de peliculas

Trabajo final para la materia de Inteligencia Artificial de la carrera de Ingeniería en Computación de la UNRN.

## Consigna del Trabajo Práctico Final

Proponer un proyecto usando algún algoritmo de machine learning. El proyecto de IA debe ser de aplicación, es decir, una solución práctica a un problema real o ficticio. El mismo debe tener en cuenta la teoría vista en clase.

Una vez finalizado el proyecto, se deberán entregar los siguientes documentos:

a) **Programas fuentes** indicando el framework utilizado y el dataset, describiendo los pasos usados para limpiar el mismo.

b) **Video explicando la funcionalidad** del desarrollo, es decir, su funcionamiento. Dicha exposición deberá ser acompañada mediante un PPT donde se explicará qué parte de la teoría se usó y se mostrará en el código fuente.

c) **Presentar las métricas** que avalen los resultados que se obtienen.

d) Es un **trabajo individual**.

e) **Fecha de entrega TP Final:** 29 de octubre de 2025. **Recuperación de entrega:** 12 de noviembre de 2025.

-----

## Dataset Utilizado

Para este proyecto, se utilizaron dos archivos CSV descargados de MovieLens:

  * **`ratings.csv`**: Tiene las valoraciones que los usuarios fueron dando a las películas. Originalmente con 3 millones de entradas, las filtre para trabajar con los 1000 usuarios más activos y las 1000 películas más valoradas para optimizar el rendimiento.

      * **Estructura**: `userId`, `movieId`, `rating`, `timestamp`

  * **`movies.csv`**: Contiene la información de las películas. El dataset original cuenta con 90 mil películas.

      * **Estructura**: `movieId`, `title`, `genres`

-----

## Código y Desarrollo

El sistema de recomendación se desarrolló en Python utilizando las librerías **Pandas** para la manipulación de datos y **Scikit-learn** para el cálculo de la similitud del coseno.

El algoritmo es un **sistema de recomendación basado en filtrado colaborativo item-item**. Este programa recomienda películas a los usuarios basándose en la similitud entre las películas que fueron valoradas positivamente en el pasado y otras películas del catálogo.

A continuación el código principal del proyecto:

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Carga de datasets
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Reducción del tamaño del dataset para optimizar el rendimiento (El dataset tenia muchas entradas)
top_movies = ratings['movieId'].value_counts().head(1000).index
top_users = ratings['userId'].value_counts().head(1000).index
ratings_small = ratings[ratings['movieId'].isin(top_movies) & ratings['userId'].isin(top_users)]

# Creación de la matriz usuario-ítem
user_item_matrix = ratings_small.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Cálculo de la similitud del coseno entre películas
cosine_sim = cosine_similarity(user_item_matrix.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recomendar_peliculas(movie_title, movies_df, similarity_df, top_n=5):
    """
    Función que recomienda películas basadas en la similitud del coseno.
    """
    movie_id = movies_df[movies_df['title'] == movie_title]['movieId'].values
    if len(movie_id) == 0:
        return f"La película '{movie_title}' no se encontró en el dataset."
    
    movie_id = movie_id[0]
    
    # Obtener películas similares y ordenarlas
    similar_movies = similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_movies.drop(movie_id) # Excluir la película elegida

    # Obtener los títulos de las películas más similares
    top_movies = similar_movies.head(top_n).index
    top_movie_titles = movies_df[movies_df['movieId'].isin(top_movies)]['title'].values
    
    return list(top_movie_titles)

# Ejemplo de uso
pelicula = "Star Wars: Episode IV - A New Hope (1977)"
recomendadas = recomendar_peliculas(pelicula, movies, cosine_sim_df)
print(f"Películas recomendadas para '{pelicula}':\n", recomendadas)
```

-----

## Resultados y Métricas

En esta sección se van a presentar las métricas utilizadas para evaluar el rendimiento del sistema de recomendación. *(agregar las métricas, por ejemplo, Precisión, Recall, F1-Score, etc., y los resultados obtenidos)*.

-----

## Video y Presentación

  * **Video Explicativo**: [Link al video] 
  * **Presentación PPT**: [Link a la presentación]
  * **Link del dataset**: https://grouplens.org/datasets/movielens/

-----

**Autor**: Galo Orellana