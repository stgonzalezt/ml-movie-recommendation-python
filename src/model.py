# src/model.py
"""
Modelo de Recomendación utilizando Surprise

Este script carga datos de calificaciones de películas, entrena un modelo SVD (Singular Value Decomposition)
utilizando la biblioteca Surprise, y evalúa el rendimiento del modelo utilizando RMSE (Root Mean Squared Error).

Dependencias:
- surprise
- pandas

Autor:
stgonzalezt

Versión:
1.0
"""
import os
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib


def load_data():
    """
    - Carga datos de calificaciones de películas desde un archivo CSV.
    - Utiliza la biblioteca Surprise para transformar estos datos en un formato compatible con Dataset, 
    que es utilizado por los algoritmos de recomendación de Surprise.

    Returns:
        surprise.DatasetAutoFolds: Conjunto de datos en formato Surprise Dataset.
    """
    #ratings = pd.read_csv('../data/processed/ratings_cleaned.csv')
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/ratings_cleaned.csv'))
    ratings = pd.read_csv(filepath)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    return data

def train_model(data):
    """
    - Divide los datos en conjuntos de entrenamiento y prueba (trainset y testset) utilizando  la función train_test_split de Surprise.
    - Entrena un modelo de recomendación utilizando el algoritmo SVD (Descomposición en Valores Singulares) sobre el conjunto de entrenamiento (trainset).

    Args:
        data (surprise.DatasetAutoFolds): Conjunto de datos en formato Surprise Dataset.

    Returns:
        tuple: Un par (modelo entrenado, conjunto de prueba).
    """
    trainset, testset = train_test_split(data, test_size=0.25)
    modelTrained = SVD()
    modelTrained.fit(trainset)

    return modelTrained, testset

def evaluate_model(modelTrained, testset):
    """
    - Evalúa el rendimiento del modelo utilizando el conjunto de prueba (testset).
    - La evaluación se realiza calculando el error RMSE (Root Mean Squared Error) 
    entre las predicciones del modelo y las calificaciones reales del conjunto de prueba.
    - Muestra el RMSE (Root Mean Squared Error/ Error Cuadrático Medio).

    Args:
        modelTrained (surprise.prediction_algorithms.matrix_factorization.SVD): Modelo entrenado.
        testset (list of tuples): Conjunto de prueba en formato Surprise.

    """
    predictions = modelTrained.test(testset)
    rmse = accuracy.rmse(predictions)
    return rmse

def save_model(model, filename='trained_model.pkl'):
    """
    Utiliza joblib.dump() para guardar el modelo en un archivo trained_model.pkl en el directorio actual.
    """
    joblib.dump(model, filename)    

if __name__ == "__main__":
    # Carga el conjunto de datos utilizando load_data()
    data = load_data()

    # Entrena un modelo llamando a train_model(data) y obtiene el modelo entrenado (modelTrained) y el conjunto de prueba (testset).
    modelTrained, testset = train_model(data)

    # Evalúa el modelo entrenado utilizando evaluate_model(modelTrained, testset) para calcular y mostrar el RMSE.
    evaluate_model(modelTrained, testset)

    save_model(modelTrained)