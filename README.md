# Prediccion de peliculas

Trabajo final para la materia de Inteligencia Artificial de la carrera de Ingeniería en Computación de la UNRN.

## Descripción del Proyecto

Este proyecto implementa un **sistema de recomendación de películas** utilizando **filtrado colaborativo item-item**. El algoritmo recomienda películas a los usuarios basándose en la similitud con otras películas que fueron valorado positivamente en el pasado por otros usuarios.

La similitud entre películas se calcula utilizando la métrica de **similitud del coseno**, que determina qué tan parecidas son dos películas en función de las calificaciones que fueron recibiendo de los mismos usuarios a traves del tiempo.

-----

## Dataset Utilizado

Para este proyecto, se utilizaron dos archivos CSV del dataset de [MovieLens](https://www.google.com/search?q=https://groulens.org/datasets/movielens/):

  * **`ratings.csv`**: Tiene millones de calificaciones (`userId`, `movieId`, `rating`) que los usuarios han dado a las películas.
  * **`movies.csv`**: Tiene la información de las películas (`movieId`, `title`, `genres`).

Para optimizar el rendimiento computacional, el dataset fue filtrado para trabajar con un subconjunto de los **1000 usuarios más activos** y las **1000 películas más valoradas**.

-----

## Frameworks Utilizados

  * **Python 3**
  * **Pandas**: Para la manipulación y análisis de datos.
  * **Scikit-learn**: Para el cálculo de la similitud del coseno.
  * **Pickle**: Para serializar y guardar el modelo entrenado.

-----

## Estructura del Proyecto

El proyecto está organizado en tres scripts principales para separar las fases de entrenamiento, evaluación y uso práctico:

  * **`entrenar_modelo.py`**:

      * **Propósito**: Realizar el trabajo pesado de procesar los datos, construir la matriz usuario-película y calcular la matriz de similitud.
      * **Resultado**: Guarda el modelo entrenado (`modelo_similitud.pkl`) y el DataFrame de películas (`movies_df.pkl`) para su uso posterior. Se ejecuta una sola vez.

  * **`evaluar_modelo.py`**:

      * **Propósito**: Realizar una evaluación académica del rendimiento del modelo. Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%) para calcular las métricas de rendimiento.
      * **Resultado**: Imprime en consola la **Precisión@10** y el **Recall@10** del modelo.

  * **`recomendar_peliculas.py`**:

      * **Propósito**: Es la aplicación final. Carga el modelo pre-entrenado y genera recomendaciones de películas de forma rápida para un título específico.
      * **Resultado**: Muestra una lista de 5 películas recomendadas.

-----

## Instalación y Uso

### 1\. Prerrequisitos

Hay que tener Python 3 instalado. Despues, instala las librerías necesarias:

```bash
pip install pandas scikit-learn
```

### 2\. Paso 1: Entrenar el Modelo

Ejecuta este script primero. Va a procesar los datos y genera los archivos del modelo.

```bash
python entrenar_modelo.py
```

### 3\. Paso 2: Obtener Recomendaciones

Una vez que el modelo está entrenado, podes ejecutar este script para obtener recomendaciones.

```bash
python recomendar_peliculas.py
```

Por defecto, tiene recomendaciones para "Star Wars: Episode IV - A New Hope (1977)". Podes editar el archivo para cambiar la película.

```bash 
    pelicula_ejemplo = "Star Wars: Episode IV - A New Hope (1977)"
```

-----

## Evaluación y Métricas 📊

Para validar el rendimiento del modelo, se uso el script `evaluar_modelo.py`. Los resultados obtenidos son los siguientes:

  * **Precisión @10**: `0.0852`
  * **Recall @10**: `0.0174`

### Significado de las Métricas

  * **Precisión**: Indica que de cada 10 películas que el sistema recomienda, aproximadamente el **8.5%** fueron relevantes para el usuario (es decir, casi 1 de cada 10 le acerto). Mide la **calidad** y importancia de las recomendaciones.

  * **Recall**: Muestra que el modelo fue capaz de encontrar el **1.7%** del total de películas que a un usuario le podrían haber gustado de todo el catálogo. Mide la **cobertura** y el alcance del modelo para descubrir ítems relevantes.

Estos valores son esperables para un sistema de recomendación de filtrado colaborativo, y queda demostrado que el modelo captura patrones de preferencia mejores que el azar.

Para obtener estos mismos resultados, ejecuta:

```bash
python evaluar_modelo.py
```

-----

## Video y Presentación

  * **Video Explicativo**: [Link al video]
  * **Presentación PPT**: [Link a la presentación]

-----

**Autor**: Galo Orellana