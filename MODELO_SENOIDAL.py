# modelo_senoidal.py
# Autor: Daniel Cano Duque
# Proyecto PyTorch 2025 – Entregable 1

try:
    import torch
    import torch.nn as nn
except (ModuleNotFoundError, ImportError) as e:
    raise ImportError("PyTorch no está instalado.") from e # Instrucciones de instalación en https://pytorch.org/get-started/locally/

import numpy as np
import matplotlib.pyplot as plt

class ModeloSenoidal:
    """
    Clase para generar datos, construir, entrenar y visualizar un modelo 
    que aprende la relación y = sin(x).
    """

    def __init__(self): # Inicialización de variables
        self.modelo = None # Modelo de red neuronal
        self.x_entrenamiento = None # Datos de entrada para entrenamiento
        self.y_entrenamiento = None # Etiquetas para entrenamiento
        self.x_validacion = None # Datos de entrada para validación
        self.y_validacion = None # Etiquetas para validación
        self.historial_perdidas = [] # Registro de pérdidas durante entrenamiento

    def generar_datos(self, n_samples: int = 1000, rango: tuple = (0, 2 * np.pi)): # Generar datos de entrenamiento y validación
        """
        Genera datos y=sin(x) mezclados aleatoriamente para evitar sobreajuste
        por extrapolación de rango.
        """
        x = np.linspace(rango[0], rango[1], n_samples) # Datos de entrada
        y = np.sin(x) # Etiquetas correspondientes  

        # Convertir a tensores
        x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1) 

        # --- CORRECCIÓN DE SOBREAJUSTE ---
        # Mezclamos los índices aleatoriamente usando randperm.
        # Esto asegura que tanto entrenamiento como validación tengan datos de todo el rango.
        indices = torch.randperm(n_samples) # Índices mezclados
        split = int(0.8 * n_samples) # 80% para entrenamiento, 20% para validación
        
        idx_train = indices[:split] # Índices para entrenamiento
        idx_val = indices[split:] # Índices para validación

        # Asignamos usando los índices mezclados
        self.x_entrenamiento = x_tensor[idx_train] # Datos de entrada para entrenamiento
        self.y_entrenamiento = y_tensor[idx_train] # Etiquetas para entrenamiento
        self.x_validacion = x_tensor[idx_val] # Datos de entrada para validación
        self.y_validacion = y_tensor[idx_val] # Etiquetas para validación

        return self.x_entrenamiento, self.y_entrenamiento, self.x_validacion, self.y_validacion #| Retornar datos generados
    # --- INCISO 2: CONSTRUIR Y ENTRENAR MODELO  ---

    def construir_modelo(self):
        """
        Red neuronal simple. Se aumentó ligeramente la capacidad para mejor ajuste.
        """
        self.modelo = nn.Sequential(
            nn.Linear(1, 20),   # Aumentado a 20 neuronas para mejor "suavidad"
            nn.Tanh(),          # Tanh es excelente para funciones como seno
            nn.Linear(20, 1)   # Capa de salida
        )
        return self.modelo #| Retornar modelo construido

    def entrenar(self, epochs: int = 2000, lr: float = 0.01): # Entrenar el modelo
        """
        Ciclo de entrenamiento. Se aumentaron los epochs por defecto para mejor convergencia.
        """
        if self.modelo is None or self.x_entrenamiento is None: # Verificación previa
            raise ValueError("Debe construir el modelo y generar datos primero.") #| Error si no se ha construido el modelo o generado datos

        criterion = nn.MSELoss() # Función de pérdida: Error Cuadrático Medio
        optimizer = torch.optim.Adam(self.modelo.parameters(), lr=lr) # Optimizador Adam
        self.historial_perdidas = [] # Reiniciar historial de pérdidas

        print(f"\nIniciando entrenamiento por {epochs} épocas...") 
        
        for epoch in range(epochs): # Ciclo de entrenamiento
            # 1. Forward
            y_predicho = self.modelo(self.x_entrenamiento) # Predicciones del modelo
            
            # 2. Loss
            loss = criterion(y_predicho, self.y_entrenamiento) # Cálculo de pérdida
            
            # 3. Backward
            optimizer.zero_grad() # Reiniciar gradientes
            loss.backward() # Propagación hacia atrás
            optimizer.step() # Actualizar parámetros
            
            self.historial_perdidas.append(loss.item()) # Registrar pérdida

            if (epoch + 1) % 200 == 0:  # Reportar cada 200 épocas para no saturar consola
                with torch.no_grad():
                    y_val_pred = self.modelo(self.x_validacion) # Predicciones de validación
                    loss_val = criterion(y_val_pred, self.y_validacion) # Pérdida de validación
                print(f"Época [{epoch + 1}/{epochs}] | Loss Train: {loss.item():.6f} | Loss Val: {loss_val.item():.6f}") # Reporte de pérdida

        print("✓ Entrenamiento completado.")

    def predecir(self, x):
        if self.modelo is None:
            raise ValueError("Modelo no construido.")
        
        if isinstance(x, np.ndarray): # Convertir numpy a tensor si es necesario
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1) # Asegurar forma correcta
            
        with torch.no_grad(): # No calcular gradientes durante la predicción
            predicciones = self.modelo(x) # Obtener predicciones
        return predicciones

    # --- INCISO 3: GRAFICAR RESULTADOS  ---
    def graficar_resultados(self, nombre_archivo="prediccion_vs_real.png"):
        """
        Genera y guarda las gráficas de comparación y pérdida.
        """
        if self.modelo is None:
            return

        # Generamos un rango ordenado para que la línea se dibuje suavemente
        x_test = np.linspace(0, 2*np.pi, 200) # Datos de prueba ordenados
        y_test_real = np.sin(x_test) # Valores reales de la función seno
        y_test_pred = self.predecir(x_test).numpy() # Predicciones del modelo

        plt.figure(figsize=(12, 5))# Tamaño más amplio para dos subplots

        # Subplot 1: Predicción vs Real
        plt.subplot(1, 2, 1) # Predicción vs Real
        plt.plot(x_test, y_test_real, label="Real (sin(x))", color="blue", alpha=0.6, linewidth=2) # Línea más gruesa para mejor visibilidad
        plt.plot(x_test, y_test_pred, label="Red Neuronal", color="red", linestyle="--", linewidth=2) # Línea discontinua para distinguir
        # Graficamos también algunos puntos de entrenamiento para ver cobertura de datos    
        plt.scatter(self.x_entrenamiento.numpy(), self.y_entrenamiento.numpy(), 
                    s=5, color="gray", alpha=0.3, label="Datos Entrenamiento")# Puntos de datos de entrenamiento
        
        plt.title("Ajuste de Función Senoidal")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Curva de Pérdida (Loss vs Epoch) 
        plt.subplot(1, 2, 2) # Pérdida en escala logarítmica
        plt.plot(self.historial_perdidas, label="MSE Loss", color="purple") # Color más visible
        plt.title("Evolución de la Pérdida durante Entrenamiento") # Título descriptivo
        plt.xlabel("Épocas") # Número de épocas en el eje x
        plt.ylabel("Pérdida (MSE)") # Pérdida en el eje y  
        plt.yscale("log") # Escala logarítmica para ver mejor la convergencia final
        plt.grid(True, alpha=0.3)  # Cuadrícula más sutil


        plt.tight_layout()  # Ajustar subplots para evitar solapamientos
        plt.savefig(nombre_archivo) # Guardar figura
        print(f"✓ Gráfica guardada como '{nombre_archivo}'")
        plt.show()

    # --- INCISO EXTRA: GUARDAR MODELO (Tabla 1 - Archivos)  ---
    def guardar_modelo(self, path="modelo_senoidal.pth"):# Guardar parámetros del modelo entrenado
        torch.save(self.modelo.state_dict(), path) 
        print(f"✓ Modelo guardado en '{path}'") 


if __name__ == "__main__": 
    print("=" * 60)
    print("PROYECTO PYTORCH - ENTREGABLE 1") # Título del proyecto
    print("Autor: Daniel Cano Duque")  # Autor del proyecto 
    print("=" * 60)
    
    proy = ModeloSenoidal() # Instanciar clase del proyecto
    
    # 1. Generar datos (ahora con mezcla aleatoria)
    proy.generar_datos()
    
    # 2. Construir
    proy.construir_modelo()
    
    # 3. Entrenar (más épocas para mejor ajuste)
    proy.entrenar(epochs=2000, lr=0.01) 
    
    # 4. Graficar (Completa instrucción 3 y entregable png)
    proy.graficar_resultados()
    
    # 5. Guardar parámetros (Completa etapa 3 de Tabla 1)
    proy.guardar_modelo()