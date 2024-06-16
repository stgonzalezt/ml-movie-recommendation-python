# tests/test_model.py
import unittest
from src.model import load_data, train_model, evaluate_model

class TestModel(unittest.TestCase):
    """
    Caso de prueba para el modelo de recomendación de películas.
    """

    def setUp(self):
        """
        Configura el entorno de prueba.
        Carga los datos y entrena el modelo antes de cada prueba.
        """
        self.data = load_data()
        self.modelTrained, self.testset = train_model(self.data)

    def test_load_data(self):
        """
        Prueba que los datos se carguen correctamente.
        Verifica que los datos no sean None.
        """
        self.assertIsNotNone(self.data)

    def test_train_model(self):
        """
        Prueba que el modelo se entrene correctamente.
        Verifica que el modelo entrenado no sea None y que el conjunto de prueba no esté vacío.
        """
        self.assertIsNotNone(self.modelTrained)
        self.assertGreater(len(self.testset), 0)

    def test_evaluate_model(self):
        """
        Prueba que la evaluación del modelo devuelva un RMSE válido.
        Verifica que el RMSE sea un float y que sea mayor o igual a 0.
        """
        rmse = evaluate_model(self.modelTrained, self.testset)
        self.assertIsInstance(rmse, float)
        self.assertGreaterEqual(rmse, 0)

if __name__ == "__main__":
    unittest.main()
