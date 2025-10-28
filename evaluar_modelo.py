import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Iniciando la evaluación del modelo...")

try:
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
except FileNotFoundError:
    print("Error: Asegúrate de que 'ratings.csv' y 'movies.csv' estén en la misma carpeta.")
    exit()

top_movies = ratings['movieId'].value_counts().head(1000).index
top_users = ratings['userId'].value_counts().head(1000).index
ratings_small = ratings[ratings['movieId'].isin(top_movies) & ratings['userId'].isin(top_users)]
print("Datos cargados y filtrados.")

train_data, test_data = train_test_split(
    ratings_small,
    test_size=0.20,
    random_state=42,
    stratify=ratings_small['userId']
)

print(f"Datos de entrenamiento: {len(train_data)} filas")
print(f"Datos de prueba: {len(test_data)} filas")
print("-" * 40)

user_item_matrix_train = train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
print("Matriz usuario-película de entrenamiento creada.")

cosine_sim_train = cosine_similarity(user_item_matrix_train.T)
cosine_sim_df_train = pd.DataFrame(
    cosine_sim_train,
    index=user_item_matrix_train.columns,
    columns=user_item_matrix_train.columns
)
print("Matriz de similitud del coseno calculada.")

def obtener_recomendaciones_ids(movie_id, similarity_df, top_n=10):
    """Obtiene una lista de IDs de películas recomendadas."""
    if movie_id not in similarity_df.index:
        return []

    similar_movies = similarity_df[movie_id].sort_values(ascending=False)
    similar_movies = similar_movies.drop(movie_id)
    top_movies_ids = similar_movies.head(top_n).index
    return list(top_movies_ids)

k = 10
precisions = []
recalls = []

print(f"\nCalculando métricas para {test_data['userId'].nunique()} usuarios en el set de prueba...")

for user_id, group in test_data.groupby('userId'):
    train_movies_rated_by_user = train_data[train_data['userId'] == user_id]
    
    if train_movies_rated_by_user.empty:
        continue

    last_movie_id = train_movies_rated_by_user.sort_values('timestamp', ascending=False).iloc[0]['movieId']
    
    recommendations_ids = obtener_recomendaciones_ids(last_movie_id, cosine_sim_df_train, top_n=k)
    
    if not recommendations_ids:
        continue

    relevant_items_in_test = group[group['rating'] >= 4.0]['movieId'].tolist()

    if not relevant_items_in_test:
        continue

    hits = len(set(recommendations_ids).intersection(set(relevant_items_in_test)))

    precision = hits / k
    recall = hits / len(relevant_items_in_test)
    
    precisions.append(precision)
    recalls.append(recall)

average_precision = np.mean(precisions) if precisions else 0
average_recall = np.mean(recalls) if recalls else 0

os.makedirs("graficas", exist_ok=True)

# Histograma de precisión
plt.figure(figsize=(8, 5))
sns.histplot(precisions, bins=20, color='skyblue', edgecolor='black')
plt.title(f"Distribución de Precisión @{k}")
plt.xlabel("Precisión")
plt.ylabel("Cantidad de usuarios")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/hist_precision.png")
plt.close()

# Histograma de recall
plt.figure(figsize=(8, 5))
sns.histplot(recalls, bins=20, color='salmon', edgecolor='black')
plt.title(f"Distribución de Recall @{k}")
plt.xlabel("Recall")
plt.ylabel("Cantidad de usuarios")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/hist_recall.png")
plt.close()

# Boxplots comparativos
plt.figure(figsize=(6, 5))
sns.boxplot(data=[precisions, recalls], palette=["skyblue", "salmon"])
plt.xticks([0, 1], [f"Precisión @{k}", f"Recall @{k}"])
plt.title("Distribución de métricas por usuario")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/boxplot_metrics.png")
plt.close()

# Curva promedio Precisión–Recall
plt.figure(figsize=(6, 5))
plt.scatter(recalls, precisions, color="purple", alpha=0.6, label="Usuarios")
plt.axhline(average_precision, color='blue', linestyle='--', label='Precisión promedio')
plt.axvline(average_recall, color='red', linestyle='--', label='Recall promedio')
plt.title(f"Curva Precisión–Recall @{k}")
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graficas/precision_recall_curve.png")
plt.close()

print("-" * 40)
print("✅ EVALUACIÓN COMPLETADA")
print("-" * 40)
print("--- MÉTRICAS PROMEDIO DEL MODELO ---")
print(f"Precisión @{k}: {average_precision:.4f}")
print(f"Recall @{k}:    {average_recall:.4f}")
print("-" * 40)
print("\n**Definiciones:**")
print(f"**Precisión**: De cada {k} películas recomendadas, {average_precision:.1%} fueron relevantes para el usuario.")
print(f"**Recall**: El modelo logró encontrar el {average_recall:.1%} del total de películas relevantes para el usuario.")
print("\n✅ Gráficas generadas y guardadas en la carpeta 'graficas/'")