# modelo_senoidal.py
# Autor: Daniel Cano Duque
# Proyecto PyTorch 2025 – Entregable 1
# Inciso 1: Fundamentos – Clases, métodos y construcción de una red neuronal simple
import torch # Importa la librería principal de PyTorch para trabajar con tensores y redes neuronales
import torch.nn as nn # Importa el módulo de redes neuronales de PyTorch
import numpy as np # Importa NumPy para manejo de arrays y funciones matemáticas
import matplotlib.pyplot as plt # Importa Matplotlib para visualización de datos

#definición de la clase ModeloSenoidal
class ModeloSenoidal:
    """
    Clase base para generar datos y construir un modelo neuronal
    que aprenda la relación y = sin(x).
    """

    def __init__(self):
        # Inicializa los atributos del modelo
        self.modelo = None
        self.x_entrenamiento = None
        self.y_entrenamiento = None

    def generar_datos(self, n_samples: int = 1000, rango: tuple = (0, 2 * np.pi)):
        """
        Genera datos (x, y) donde y = sin(x), dentro del rango dado.
        Los datos se convierten a tensores de PyTorch.
        """
        x = np.linspace(rango[0], rango[1], n_samples)
        y = np.sin(x)

        # Convertimos a tensores
        self.x_entrenamiento = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        self.y_entrenamiento = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        return self.x_entrenamiento, self.y_entrenamiento

    def construir_modelo(self):
        """
        Crea una red neuronal simple con una capa oculta y activación Tanh.
        """
        self.modelo = nn.Sequential(
            nn.Linear(1, 10),  # capa de entrada -> oculta
            nn.Tanh(),
            nn.Linear(10, 1)   # capa de salida
        )

        return self.modelo


# Bloque de prueba
if __name__ == "__main__":
    modelo = ModeloSenoidal()
    x, y = modelo.generar_datos()
    red = modelo.construir_modelo()

    print("Datos generados:", x.shape, y.shape)
    print("Estructura del modelo:")
    print(red)
