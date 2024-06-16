# src/data_processing.py
"""
Script de Procesamiento de Datos: Limpia, transforma y guarda datos relacionados con un sistema de recomendación de 
películas utilizando datos de MovieLens.

Dependencias:
- pandas
- os

Autor:
stgonzalezt

Versión:
1.0
"""
import pandas as pd
import os

def clean_data(movies, ratings, tags):
    """
    Limpia los datos de entrada eliminando duplicados y manejando valores nulos.

    Args:
        movies (pd.DataFrame): DataFrame que contiene la información de las películas.
        ratings (pd.DataFrame): DataFrame que contiene las calificaciones de las películas.
        tags (pd.DataFrame): DataFrame que contiene las etiquetas asociadas a las películas.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames de películas, calificaciones y etiquetas limpios.
    """
    movies.drop_duplicates(inplace=True)
    ratings.drop_duplicates(inplace=True)
    tags.drop_duplicates(inplace=True)

    movies.dropna(inplace=True)
    ratings.dropna(inplace=True)
    tags.dropna(inplace=True)
    
    return movies, ratings, tags

def transform_data(movies, ratings, tags):
    """
    Transforma los datos de entrada convirtiendo los campos de fecha al formato adecuado.

    Args:
        movies (pd.DataFrame): DataFrame que contiene la información de las películas.
        ratings (pd.DataFrame): DataFrame que contiene las calificaciones de las películas.
        tags (pd.DataFrame): DataFrame que contiene las etiquetas asociadas a las películas.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames de películas, calificaciones y etiquetas transformados.
    """
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
    
    return movies, ratings, tags

def save_data(movies, ratings, tags, output_path):
    """
    Guarda los datos procesados en archivos CSV en la ruta especificada.

    Args:
        movies (pd.DataFrame): DataFrame que contiene la información de las películas limpias y transformadas.
        ratings (pd.DataFrame): DataFrame que contiene las calificaciones de las películas limpias y transformadas.
        tags (pd.DataFrame): DataFrame que contiene las etiquetas asociadas a las películas limpias y transformadas.
        output_path (str): Ruta del directorio donde se guardarán los archivos CSV procesados.
    """
    # Construcción de rutas de salida utilizando os.path.join
    movies_output_path = os.path.join(output_path, 'movies_cleaned.csv')
    ratings_output_path = os.path.join(output_path, 'ratings_cleaned.csv')
    tags_output_path = os.path.join(output_path, 'tags_cleaned.csv')

    # Guardar los datos procesados en archivos CSV
    movies.to_csv(movies_output_path, index=False)
    ratings.to_csv(ratings_output_path, index=False)
    tags.to_csv(tags_output_path, index=False)

if __name__ == "__main__":
    """
    Función principal que ejecuta el flujo completo de carga, limpieza, transformación y guardado de los datos.
    """
    # Definir las rutas de los datos
    data_path = '../data/raw'
    output_path = '../data/processed'
    
    # Cargar los datos directamente en el main
    movies_path = os.path.join(data_path, 'movies.csv')
    ratings_path = os.path.join(data_path, 'ratings.csv')
    tags_path = os.path.join(data_path, 'tags.csv')
    
    # Cargar los diferentes archivos de MovieLens
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)
    
    # Limpiar los datos
    movies_cleaned, ratings_cleaned, tags_cleaned = clean_data(movies, ratings, tags)
    
    # Transformar los datos
    movies_transformed, ratings_transformed, tags_transformed = transform_data(movies_cleaned, ratings_cleaned, tags_cleaned)
    
    # Crear el directorio de salida si no existe
    os.makedirs(output_path, exist_ok=True)
    
    # Guardar los datos procesados
    save_data(movies_transformed, ratings_transformed, tags_transformed, output_path)
    
    print("Datos procesados y guardados correctamente.")
