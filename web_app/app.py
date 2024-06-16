# web_app/app.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
import pandas as pd
from src.recommend import recommend, load_model

app = Flask(__name__)

# Cargar los datos y el modelo
movies = pd.read_csv('../data/processed/movies_cleaned.csv')
algo = load_model()

@app.route('/')
def home():
    """
    Renderiza la página de inicio.
    Returns:
        str: El contenido HTML de la página de inicio.
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    """
    Genera y muestra recomendaciones de películas para un usuario dado.
    Lee el ID del usuario del formulario, obtiene las recomendaciones y renderiza la página de recomendaciones.

    Returns:
        str: El contenido HTML de la página de recomendaciones con las películas recomendadas.
    """
    user_id = int(request.form['user_id'])
    recommendations = recommend(algo, user_id, movies)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
