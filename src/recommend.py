# src/recommend.py
"""
Script para recomendar películas utilizando un modelo de recomendación entrenado.

Este script carga datos de calificaciones limpios, un modelo preentrenado y una lista de películas.
Valida el rendimiento del modelo y genera recomendaciones personalizadas de películas para un usuario.

Dependencias:
- pandas
- surprise

Autor:
stgonzalezt

Versión:
1.0
"""

import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib
from surprise import accuracy

def load_data():
    """
    Carga los datos de calificaciones limpios (ratings_cleaned.csv) y los convierte en un objeto Dataset de Surprise.

    Returns:
        surprise.dataset.DatasetAutoFolds: Objeto Dataset que contiene los datos de calificaciones.
    """
    ratings = pd.read_csv('../data/processed/ratings_cleaned.csv')
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    return data

def load_model(model_filename='../src/trained_model.pkl'):
    """
    Carga un modelo preentrenado desde el archivo trained_model.pkl utilizando joblib.

    Args:
        model_filename (str): Ruta del archivo del modelo preentrenado.

    Returns:
        surprise.prediction_algorithms.matrix_factorization.SVD: Modelo SVD cargado.
    """
    modelTrained = SVD()
    modelTrained = joblib.load(model_filename)

    return modelTrained

def validate_model(modelTrained, testset):
    """
    Valida el modelo entrenado utilizando un conjunto de prueba (testset) y calcula el RMSE.

    Args:
        modelTrained (surprise.prediction_algorithms.matrix_factorization.SVD): Modelo SVD entrenado.
        testset (list of tuples): Conjunto de prueba para la validación.

    Returns:
        float: Error cuadrático medio (RMSE) de las predicciones del modelo.
    """
    predictions = modelTrained.test(testset)
    rmse = accuracy.rmse(predictions)

    return rmse

def recommend(modelTrained, user_id, movies, n=10):
    """
    Genera 10 recomendaciones de películas para un usuario dado utilizando el modelo entrenado.

    Args:
        modelTrained (surprise.prediction_algorithms.matrix_factorization.SVD): Modelo SVD entrenado.
        user_id (int): ID de usuario para quien se generarán las recomendaciones.
        movies (pandas.DataFrame): DataFrame que contiene datos de películas.

    Returns:
        pandas.DataFrame: DataFrame que contiene las películas recomendadas.
    """
    # Obtener los IDs de películas que están presentes en el DataFrame movies.
    movie_ids = movies['movieId'].unique()

    # Para cada ID de película en movie_ids, se utiliza el modelo entrenado (modelTrained) para predecir la calificación que el usuario (user_id) daría a esa película (movie_id). 
    # Este devuelve la estimación de la calificación.
    user_ratings = [(movie_id, modelTrained.predict(user_id, movie_id).est) for movie_id in movie_ids]

    # Las predicciones de calificación (user_ratings) se ordenan en orden descendente basado en la estimación de la calificación (x[1]).
    user_ratings = sorted(user_ratings, key=lambda x: x[1], reverse=True)

    #Se seleccionan los IDs de las películas recomendadas basadas en las n mejores predicciones de calificación.
    recommended_movie_ids = [movie_id for movie_id, rating in user_ratings[:n]]

    # Se filtra el DataFrame movies para obtener las filas de las películas recomendadas, aquellas cuyos IDs están en recommended_movie_ids.
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]

    # Devuelve un DataFrame que contiene las películas recomendadas para el usuario especificado.
    return recommended_movies.to_dict('records')

if __name__ == "__main__":
    data = load_data()
    modelTrained = load_model()

    # Validar el modelo con datos de prueba
    testset = data.build_full_trainset().build_testset()
    rmse = validate_model(modelTrained, testset)
          
    user_id = 78  # Ejemplo de usuario
    movies = pd.read_csv('../data/processed/movies_cleaned.csv')
    recommendations = recommend(modelTrained, user_id, movies)
    print(recommendations)