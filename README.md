# Table of Contents
- [Table of Contents](#table-of-contents)
    - [Nombre del proyecto](#nombre-del-proyecto)
    - [GDescripción](#gdescripción)
    - [Tecnologías necesarias](#tecnologías-necesarias)
    - [Instalación](#instalación)
    - [Implementación](#implementación)
    - [Estructura del proyecto](#estructura-del-proyecto)

### Nombre del proyecto
***
Sistema de recomendación de películas: KRY-Movies

### GDescripción
***
Este proyecto implementa un sistema de recomendación de películas utilizando técnicas de filtrado colaborativo con el algoritmo SVD (Singular Value Decomposition).
 
### Tecnologías necesarias
***
Lista de las tecnologías usadas en este proyecto:
* [Python](https://www.python.org/): Version 3.12.4
* [Anaconda](https://www.anaconda.com/): Version 24.1.2
* [pandas](https://pandas.pydata.org/): Version 2.2.2
* [NumPy](https://numpy.org/): Version 1.26.4
* [Jupyter Notebook](https://jupyter.org/)
* [scikit-learn](https://scikit-learn.org/stable/): Version 1.4.2
* [Matplotlib](https://matplotlib.org): Version 3.8.4
* [Seaborn](https://seaborn.pydata.org): Version 0.13.2
* [Surprise](https://surpriselib.com): Version 1.1.4
* [Flask](https://flask.palletsprojects.com/en/3.0.x/): Version 3.0.3
* [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/): Last updated 9/2018
* [Microsoft Visual Studio (Build Tools)](https://visualstudio.microsoft.com/es/downloads/): Version 2022
* [Visual Studio Code (VS Code)](https://code.visualstudio.com): Version 1.90
* 
### Instalación
***
Clona el repositorio
```
$ git clone https://github.com/tu_usuario/tu_proyecto.git
$ cd ml-movie-recommendation-python
```
Instala las dependencias necesarias

### Implementación
***
Entrenamiento del Modelo de recomendación 
```
$ python src/model.py
```
Ejecuta la aplicación web para recibir recomendaciones personalizadas
```
$ python web_app/app.py
```

### Estructura del proyecto
ml-movie-recommendation-python/
│
├── data/
│   ├── processed/       
|   |    |──movies_cleaned.csv
|   |    |──ratings_cleaned.csv
|   |    |__tags_cleaned.csv 
|   |
|   └── raw/             
|        |──movies.csv
|        |──ratings.csv
|        |__tags.csv
|        └──links.csv
|
|── notebooks/
|        └──exploratory_data_analysis.ipynb
│
├── src/     
|   ├── __init__.py
|   ├── data_processing.py            
│   ├── model.py         
│   ├── recommend.py
|   └── trained_model.pkl 
|
├── tests/               
│   ├── __init__.py  
│   └── test_model.py     
│
├── web_app/ 
|   |── static/
|   |     ├── css/
|   |     └── images/ 
|   |                      
│   ├── templates/  
|   |     ├── index.html
|   |     └──recommendations.html
|   |
|   └── trained_model.pkl      
│   └── app.py        
│            
└── README.md           

