# modelo_senoidal.py
# Autor: Daniel Cano Duque
# Proyecto PyTorch 2025 – Entregable 1
# Etapa 2: Estructuras de datos - Implementar ciclo de entrenamiento manual

# Importa la librería principal de PyTorch para trabajar con tensores y redes neuronales
import torch

# Importa el módulo nn (neural network) de PyTorch que contiene capas y funciones de activación
import torch.nn as nn

# Importa NumPy para operaciones matemáticas y generación de arrays numéricos
import numpy as np

# Importa Matplotlib para visualizar gráficos y datos
import matplotlib.pyplot as plt

# Define una clase llamada ModeloSenoidal que encapsulará toda la funcionalidad del modelo
class ModeloSenoidal:
    """
    Clase base para generar datos y construir un modelo neuronal
    que aprenda la relación y = sin(x).
    """

    # Método constructor que se ejecuta cuando se crea una instancia de la clase
    def __init__(self):
        # Inicializa el atributo 'modelo' en None (aún no se ha construido la red neuronal)
        self.modelo = None
        # Inicializa el atributo 'x_entrenamiento' en None (aún no se han generado los datos de entrada)
        self.x_entrenamiento = None
        # Inicializa el atributo 'y_entrenamiento' en None (aún no se han generado los datos de salida)
        self.y_entrenamiento = None
        # Inicializa el atributo 'x_validacion' en None (datos para validar el modelo)
        self.x_validacion = None
        # Inicializa el atributo 'y_validacion' en None (etiquetas de validación)
        self.y_validacion = None
        # Inicializa una lista vacía para almacenar el historial de pérdidas durante el entrenamiento
        self.historial_perdidas = []

    # Define un método para generar datos sintéticos de la función seno
    # Los parámetros n_samples (número de puntos) y rango (intervalo) tienen valores por defecto
    def generar_datos(self, n_samples: int = 1000, rango: tuple = (0, 2 * np.pi)):
        """
        Genera datos (x, y) donde y = sin(x), dentro del rango dado.
        Los datos se convierten a tensores de PyTorch y se dividen en
        entrenamiento (80%) y validación (20%).
        """
        # Crea un array de n_samples valores espaciados uniformemente entre el inicio y fin del rango
        x = np.linspace(rango[0], rango[1], n_samples)
        # Calcula el seno de cada valor de x, obteniendo los valores de y
        y = np.sin(x)

        # Calcula el índice donde dividir los datos (80% para entrenamiento)
        indice_division = int(0.8 * n_samples)

        # Convierte los primeros 80% de x a tensor PyTorch y los asigna a entrenamiento
        # .view(-1, 1) reorganiza el tensor a una matriz columna (filas, 1 columna)
        self.x_entrenamiento = torch.tensor(x[:indice_division], dtype=torch.float32).view(-1, 1)
        # Convierte los primeros 80% de y a tensor PyTorch y los asigna a entrenamiento
        self.y_entrenamiento = torch.tensor(y[:indice_division], dtype=torch.float32).view(-1, 1)

        # Convierte los últimos 20% de x a tensor PyTorch y los asigna a validación
        self.x_validacion = torch.tensor(x[indice_division:], dtype=torch.float32).view(-1, 1)
        # Convierte los últimos 20% de y a tensor PyTorch y los asigna a validación
        self.y_validacion = torch.tensor(y[indice_division:], dtype=torch.float32).view(-1, 1)

        # Retorna todos los conjuntos de datos generados
        return self.x_entrenamiento, self.y_entrenamiento, self.x_validacion, self.y_validacion

    # Define un método para construir la arquitectura de la red neuronal
    def construir_modelo(self):
        """
        Crea una red neuronal simple con una capa oculta y activación Tanh.
        """
        # Crea una red neuronal secuencial (las capas se aplican una tras otra)
        self.modelo = nn.Sequential(
            # Primera capa Linear: recibe 1 entrada (valor de x) y produce 10 salidas (neuronas ocultas)
            nn.Linear(1, 10),
            # Función de activación Tanh aplicada a las 10 neuronas ocultas (introduce no-linealidad)
            nn.Tanh(),
            # Segunda capa Linear: recibe 10 entradas (de las neuronas ocultas) y produce 1 salida (valor de y)
            nn.Linear(10, 1)
        )

        # Retorna la red neuronal construida para poder usarla fuera del método
        return self.modelo

    # Define el método de entrenamiento del modelo usando ciclo manual
    def entrenar(self, epochs: int = 500, lr: float = 0.01):
        """
        Implementa el ciclo de entrenamiento manual con MSELoss y optimizador Adam.
        
        Parámetros:
        - epochs: Número de iteraciones completas sobre los datos de entrenamiento
        - lr: Learning rate (tasa de aprendizaje) que controla qué tan grandes son los pasos de actualización
        """
        # Verifica que el modelo haya sido construido antes de entrenar
        if self.modelo is None:
            # Si no existe, lanza un error informativo
            raise ValueError("Debe construir el modelo antes de entrenar. Llame a construir_modelo() primero.")
        
        # Verifica que existan datos de entrenamiento
        if self.x_entrenamiento is None or self.y_entrenamiento is None:
            # Si no existen, lanza un error informativo
            raise ValueError("Debe generar los datos antes de entrenar. Llame a generar_datos() primero.")

        # Define la función de pérdida: MSELoss (Mean Squared Error Loss)
        # Calcula el promedio de los errores cuadráticos entre predicción y valor real
        criterion = nn.MSELoss()

        # Define el optimizador Adam que ajustará los pesos del modelo
        # self.modelo.parameters() obtiene todos los pesos y sesgos de la red
        # lr es la tasa de aprendizaje que controla el tamaño de las actualizaciones
        optimizer = torch.optim.Adam(self.modelo.parameters(), lr=lr)

        # Limpia el historial de pérdidas previo (si existe)
        self.historial_perdidas = []

        # Imprime un mensaje indicando que el entrenamiento está comenzando
        print(f"\nIniciando entrenamiento por {epochs} épocas...")
        print("=" * 60)

        # CICLO DE ENTRENAMIENTO MANUAL
        # Itera desde 0 hasta epochs-1 (si epochs=500, va de 0 a 499)
        for epoch in range(epochs):
            
            # ===== PASO 1: FORWARD PASS (Propagación hacia adelante) =====
            # El modelo procesa los datos de entrada y genera predicciones
            y_predicho = self.modelo(self.x_entrenamiento)

            # ===== PASO 2: CALCULAR PÉRDIDA =====
            # Compara las predicciones con los valores reales usando MSELoss
            # Calcula: loss = (1/n) * Σ(y_predicho - y_real)²
            loss = criterion(y_predicho, self.y_entrenamiento)

            # ===== PASO 3: LIMPIAR GRADIENTES =====
            # Resetea los gradientes acumulados de iteraciones anteriores
            # Sin esto, los gradientes se acumularían y causarían errores
            optimizer.zero_grad()

            # ===== PASO 4: BACKWARD PASS (Propagación hacia atrás) =====
            # Calcula automáticamente los gradientes de la pérdida con respecto a cada peso
            # Usa backpropagation para determinar cómo ajustar cada parámetro
            loss.backward()

            # ===== PASO 5: ACTUALIZAR PESOS =====
            # Ajusta los pesos del modelo usando los gradientes calculados
            # Fórmula: peso_nuevo = peso_viejo - lr * gradiente
            optimizer.step()

            # ===== PASO 6: GUARDAR HISTORIAL =====
            # Guarda el valor de pérdida actual en la lista de historial
            # .item() convierte el tensor de pérdida en un número de Python
            self.historial_perdidas.append(loss.item())

            # ===== PASO 7: MONITOREAR PROGRESO =====
            # Cada 50 épocas, muestra el progreso del entrenamiento
            if (epoch + 1) % 50 == 0:
                # Calcula la pérdida en el conjunto de validación
                # torch.no_grad() desactiva el cálculo de gradientes (más eficiente)
                with torch.no_grad():
                    # Genera predicciones en los datos de validación
                    y_val_pred = self.modelo(self.x_validacion)
                    # Calcula la pérdida de validación
                    loss_validacion = criterion(y_val_pred, self.y_validacion)
                
                # Imprime las métricas de la época actual
                print(f"Época [{epoch + 1}/{epochs}] | "
                      f"Pérdida Entrenamiento: {loss.item():.6f} | "
                      f"Pérdida Validación: {loss_validacion.item():.6f}")

        # Imprime mensaje de finalización del entrenamiento
        print("=" * 60)
        print("✓ Entrenamiento completado exitosamente\n")

    # Define un método para realizar predicciones con el modelo entrenado
    def predecir(self, x):
        """
        Calcula las predicciones del modelo en datos nuevos.
        
        Parámetros:
        - x: Tensor de entrada o array NumPy con valores de x
        
        Retorna:
        - Tensor con las predicciones del modelo
        """
        # Verifica que el modelo haya sido construido
        if self.modelo is None:
            # Si no existe, lanza un error
            raise ValueError("Debe construir el modelo antes de predecir.")

        # Si x es un array de NumPy, conviértelo a tensor de PyTorch
        if isinstance(x, np.ndarray):
            # Convierte a tensor float32 y asegura que tenga forma (n, 1)
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1)

        # Desactiva el cálculo de gradientes (no necesario para predicción)
        # Esto hace la predicción más rápida y usa menos memoria
        with torch.no_grad():
            # Pasa los datos por el modelo para obtener predicciones
            predicciones = self.modelo(x)

        # Retorna las predicciones
        return predicciones


# Bloque que se ejecuta solo si este archivo se ejecuta directamente (no si se importa)
if __name__ == "__main__":
    # Crea una instancia (objeto) de la clase ModeloSenoidal
    print("=" * 60)
    print("PROYECTO PYTORCH - ETAPA 2: CICLO DE ENTRENAMIENTO")
    print("=" * 60)
    
    modelo = ModeloSenoidal()
    
    # Genera los datos de entrenamiento y validación
    print("\n[1/4] Generando datos...")
    x_train, y_train, x_val, y_val = modelo.generar_datos()
    print(f"✓ Datos de entrenamiento: {x_train.shape}")
    print(f"✓ Datos de validación: {x_val.shape}")
    
    # Construye la arquitectura del modelo
    print("\n[2/4] Construyendo modelo...")
    red = modelo.construir_modelo()
    print("✓ Modelo construido:")
    print(red)
    
    # Entrena el modelo
    print("\n[3/4] Entrenando modelo...")
    modelo.entrenar(epochs=500, lr=0.01)
    
    # Realiza predicciones y muestra resultados
    print("[4/4] Evaluando modelo...")
    with torch.no_grad():
        y_pred_train = modelo.predecir(x_train)
        y_pred_val = modelo.predecir(x_val)
    
    # Calcula métricas finales
    mse_train = nn.MSELoss()(y_pred_train, y_train).item()
    mse_val = nn.MSELoss()(y_pred_val, y_val).item()
    
    print(f"\n✓ MSE Final Entrenamiento: {mse_train:.6f}")
    print(f"✓ MSE Final Validación: {mse_val:.6f}")
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
    print("=" * 60)