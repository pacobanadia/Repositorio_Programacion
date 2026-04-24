# modelo_senoidal.py
# Autor: Daniel Cano Duque
# Proyecto PyTorch 2026
#
# MODIFICACIONES RESPECTO A LA VERSIÓN ANTERIOR:
#
# [1] BÚSQUEDA DE PARÁMETROS ALPHA:
#     La función de activación ahora es f(x) = alpha_1 * x^3 + alpha_2 * x,
#     donde alpha_1 y alpha_2 son parámetros que dependen de las condiciones del
#     material (en este caso, Óxido de Grafeno). Se realiza una búsqueda en cuadrícula
#     (grid search) sobre hasta 20 valores equiespaciados en [1, 5] para cada parámetro,
#     evaluando hasta 400 combinaciones (20 x 20). En cada combinación se entrena el
#     modelo completo y se registran loss_train y loss_val finales. Al terminar la
#     búsqueda, se imprimen los valores óptimos y se re-entrena el modelo final con ellos.
#
# [2] CURVA DE PÉRDIDA DUAL:
#     Se agrega el registro de loss_val por época durante el entrenamiento.
#     La gráfica de pérdida ahora muestra loss_train y loss_val superpuestas en escala
#     logarítmica, permitiendo detectar sobreajuste (divergencia entre ambas curvas).

try:
    import torch
    import torch.nn as nn
except (ModuleNotFoundError, ImportError) as e:
    raise ImportError("PyTorch no está instalado.") from e

import numpy as np
import matplotlib.pyplot as plt
import itertools  # Para generar todas las combinaciones de alpha_1 x alpha_2


# ==============================================================================
# CLASE DE ACTIVACIÓN HAMILTONIANA PARAMETRIZADA
# ==============================================================================

class HamiltonianActivation(nn.Module):
    """
    Función de activación personalizada parametrizada:

        f(x) = alpha_1 * x^3 + alpha_2 * x

    Interpretación física:
    ----------------------
    Esta función corresponde a la derivada de un Hamiltoniano efectivo de la forma:

        H(x) = (alpha_1 / 4) * x^4 + (alpha_2 / 2) * x^2 + c

    de modo que:

        dH/dx = alpha_1 * x^3 + alpha_2 * x = f(x)

    En el contexto de sensores de Óxido de Grafeno (GO), los coeficientes alpha_1
    y alpha_2 codifican propiedades del material: alpha_1 controla la no-linealidad
    cúbica (asociada a interacciones de mayor orden), mientras que alpha_2 modula
    el término lineal (análogo a una constante de restauración o rigidez efectiva).
    Buscar los valores óptimos de estos parámetros equivale a calibrar el modelo
    computacional a las condiciones físicas reales del sensor.

    Parámetros:
    -----------
    alpha_1 : float
        Coeficiente del término cúbico. Rango de búsqueda: [1, 5].
    alpha_2 : float
        Coeficiente del término lineal. Rango de búsqueda: [1, 5].
    """

    def __init__(self, alpha_1: float = 4.0, alpha_2: float = 1.0):
        super().__init__()
        # Almacenamos los parámetros como atributos simples (no como nn.Parameter)
        # porque los optimizamos fuera del grafo de PyTorch, mediante grid search.
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def forward(self, x):
        # f(x) = alpha_1 * x^3 + alpha_2 * x
        return self.alpha_1 * x**3 + self.alpha_2 * x

    def __repr__(self):
        return f"HamiltonianActivation(α₁={self.alpha_1:.4f}, α₂={self.alpha_2:.4f})"


# ==============================================================================
# CLASE PRINCIPAL DEL MODELO SENOIDAL
# ==============================================================================

class ModeloSenoidal:
    """
    Clase para generar datos, construir, entrenar y visualizar un modelo
    que aprende la relación y = sin(x), usando una activación Hamiltoniana
    parametrizada por alpha_1 y alpha_2.

    El flujo de uso recomendado es:
        1. generar_datos()
        2. buscar_parametros_optimos()   ← NUEVO: encuentra alpha_1 y alpha_2 óptimos
        3. construir_modelo(alpha_1, alpha_2)
        4. entrenar()
        5. graficar_resultados()
        6. guardar_modelo()
    """

    def __init__(self):
        self.modelo = None
        self.x_entrenamiento = None
        self.y_entrenamiento = None
        self.x_validacion = None
        self.y_validacion = None
        self.historial_loss_train = []   # [CAMBIO 2] Renombrado para claridad
        self.historial_loss_val = []     # [CAMBIO 2] NUEVO: registro de loss_val por época

    # --------------------------------------------------------------------------
    # GENERACIÓN DE DATOS
    # --------------------------------------------------------------------------

    def generar_datos(self, n_samples: int = 1000, rango: tuple = (0, 2 * np.pi)):
        """
        Genera datos y = sin(x) con partición aleatoria 80/20 para entrenamiento
        y validación. La mezcla aleatoria garantiza que ambos conjuntos cubran
        todo el rango de entrada, evitando sobreajuste por extrapolación.
        """
        x = np.linspace(rango[0], rango[1], n_samples)
        y = np.sin(x)

        x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Mezcla aleatoria para distribución uniforme del rango en ambos splits
        indices = torch.randperm(n_samples)
        split = int(0.8 * n_samples)

        idx_train = indices[:split]
        idx_val   = indices[split:]

        self.x_entrenamiento = x_tensor[idx_train]
        self.y_entrenamiento = y_tensor[idx_train]
        self.x_validacion    = x_tensor[idx_val]
        self.y_validacion    = y_tensor[idx_val]

        return (self.x_entrenamiento, self.y_entrenamiento,
                self.x_validacion,    self.y_validacion)

    # --------------------------------------------------------------------------
    # [CAMBIO 1] BÚSQUEDA EN CUADRÍCULA DE PARÁMETROS ALPHA
    # --------------------------------------------------------------------------

    def buscar_parametros_optimos(
        self,
        n_valores: int = 20,
        rango_alpha: tuple = (1.0, 5.0),
        epochs_busqueda: int = 500,
        lr: float = 0.01
    ):
        """
        Realiza una búsqueda en cuadrícula (grid search) sobre los parámetros
        alpha_1 y alpha_2 de la función de activación Hamiltoniana.

        Estrategia:
        -----------
        Se generan n_valores equiespaciados en rango_alpha para cada parámetro,
        resultando en n_valores² combinaciones totales. Para cada combinación:
          - Se construye un modelo nuevo con esa activación.
          - Se entrena durante epochs_busqueda épocas (reducidas para eficiencia).
          - Se evalúa loss_train y loss_val al final del entrenamiento.
          - Se registran los resultados en una tabla.

        Al finalizar, se imprime la tabla completa con los 10 mejores resultados
        según loss_val, y se retornan los parámetros óptimos.

        Parámetros:
        -----------
        n_valores : int
            Número de valores a probar por parámetro (máximo recomendado: 20).
            Total de evaluaciones = n_valores².
        rango_alpha : tuple
            Rango (min, max) para ambos parámetros alpha.
        epochs_busqueda : int
            Épocas de entrenamiento por combinación durante la búsqueda.
            Se usa un valor menor al entrenamiento final para reducir tiempo.
        lr : float
            Tasa de aprendizaje usada durante la búsqueda.

        Retorna:
        --------
        tuple : (mejor_alpha_1, mejor_alpha_2, resultados)
            mejor_alpha_1, mejor_alpha_2 : floats con los parámetros óptimos.
            resultados : lista de dicts con todos los resultados registrados.
        """
        if self.x_entrenamiento is None:
            raise ValueError("Debe generar datos primero con generar_datos().")

        # Generar la cuadrícula de valores para ambos parámetros
        valores_alpha = np.linspace(rango_alpha[0], rango_alpha[1], n_valores)
        combinaciones = list(itertools.product(valores_alpha, valores_alpha))
        total = len(combinaciones)

        print("=" * 70)
        print(f"BÚSQUEDA DE PARÁMETROS ALPHA — {total} combinaciones ({n_valores}×{n_valores})")
        print(f"  Rango: [{rango_alpha[0]}, {rango_alpha[1]}]  |  Épocas por prueba: {epochs_busqueda}")
        print("=" * 70)

        criterion = nn.MSELoss()
        resultados = []  # Lista para almacenar todos los resultados

        mejor_loss_val = float('inf')
        mejor_alpha_1  = None
        mejor_alpha_2  = None

        for i, (a1, a2) in enumerate(combinaciones):
            # Construir modelo temporal con esta combinación de parámetros
            modelo_temp = nn.Sequential(
                nn.Linear(1, 20),
                HamiltonianActivation(alpha_1=float(a1), alpha_2=float(a2)),
                nn.Linear(20, 1)
            )

            # Optimizador independiente para cada modelo temporal
            optimizer = torch.optim.Adam(modelo_temp.parameters(), lr=lr)

            # Entrenamiento rápido para evaluación
            for epoch in range(epochs_busqueda):
                y_pred = modelo_temp(self.x_entrenamiento)
                loss   = criterion(y_pred, self.y_entrenamiento)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluación final de esta combinación
            with torch.no_grad():
                loss_train_final = criterion(
                    modelo_temp(self.x_entrenamiento), self.y_entrenamiento
                ).item()
                loss_val_final = criterion(
                    modelo_temp(self.x_validacion), self.y_validacion
                ).item()

            # Registrar resultado
            resultado = {
                'alpha_1':    round(float(a1), 4),
                'alpha_2':    round(float(a2), 4),
                'loss_train': round(loss_train_final, 8),
                'loss_val':   round(loss_val_final,   8),
            }
            resultados.append(resultado)

            # Actualizar el mejor resultado global según loss_val
            if loss_val_final < mejor_loss_val:
                mejor_loss_val = loss_val_final
                mejor_alpha_1  = float(a1)
                mejor_alpha_2  = float(a2)

            # Reporte de progreso cada 50 combinaciones
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  [{i+1:4d}/{total}] α₁={a1:.3f}, α₂={a2:.3f} | "
                      f"loss_train={loss_train_final:.6f} | loss_val={loss_val_final:.6f}")

        # Ordenar resultados por loss_val ascendente para identificar los mejores
        resultados_ordenados = sorted(resultados, key=lambda r: r['loss_val'])

        # Imprimir tabla de los 10 mejores resultados
        print("\n" + "=" * 70)
        print("TOP 10 MEJORES COMBINACIONES (ordenadas por loss_val):")
        print(f"{'#':>3} | {'alpha_1':>8} | {'alpha_2':>8} | {'loss_train':>12} | {'loss_val':>12}")
        print("-" * 70)
        for rank, r in enumerate(resultados_ordenados[:10], start=1):
            marker = " ← ÓPTIMO" if rank == 1 else ""
            print(f"{rank:>3} | {r['alpha_1']:>8.4f} | {r['alpha_2']:>8.4f} | "
                  f"{r['loss_train']:>12.8f} | {r['loss_val']:>12.8f}{marker}")

        print("=" * 70)
        print(f"\nParámetros óptimos encontrados:")
        print(f"  α₁ (alpha_1) = {mejor_alpha_1:.4f}")
        print(f"  α₂ (alpha_2) = {mejor_alpha_2:.4f}")
        print(f"  loss_val mínimo = {mejor_loss_val:.8f}")

        return mejor_alpha_1, mejor_alpha_2, resultados_ordenados

    # --------------------------------------------------------------------------
    # CONSTRUCCIÓN DEL MODELO
    # --------------------------------------------------------------------------

    def construir_modelo(self, alpha_1: float = 4.0, alpha_2: float = 1.0):
        """
        Construye la red neuronal con la función de activación Hamiltoniana
        parametrizada por alpha_1 y alpha_2.

        Arquitectura:
            Linear(1 → 20) → HamiltonianActivation(α₁, α₂) → Linear(20 → 1)

        La capa oculta de 20 neuronas proporciona capacidad suficiente para
        aproximar la función sin(x) con suavidad en [0, 2π].
        """
        self.modelo = nn.Sequential(
            nn.Linear(1, 20),
            HamiltonianActivation(alpha_1=alpha_1, alpha_2=alpha_2),
            nn.Linear(20, 1)
        )
        print(f"\nModelo construido con activación: {self.modelo[1]}")
        return self.modelo

    # --------------------------------------------------------------------------
    # ENTRENAMIENTO CON REGISTRO DUAL DE PÉRDIDA
    # --------------------------------------------------------------------------

    def entrenar(self, epochs: int = 2000, lr: float = 0.01):
        """
        Ciclo de entrenamiento completo con registro de loss_train y loss_val
        por época.

        [CAMBIO 2] En la versión anterior solo se registraba loss_train al final
        de cada época. Ahora se evalúa loss_val en cada época (dentro de
        torch.no_grad() para no acumular gradientes innecesarios) y se almacena
        en self.historial_loss_val. Esto permite:
          - Detectar sobreajuste cuando loss_val sube mientras loss_train baja.
          - Visualizar la convergencia simultánea de ambas métricas.
          - Diagnosticar si el modelo generaliza bien fuera de los datos de entrenamiento.
        """
        if self.modelo is None or self.x_entrenamiento is None:
            raise ValueError("Debe construir el modelo y generar datos primero.")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.modelo.parameters(), lr=lr)

        # Reiniciar historiales de ambas pérdidas
        self.historial_loss_train = []
        self.historial_loss_val   = []

        print(f"\nIniciando entrenamiento final por {epochs} épocas...")

        for epoch in range(epochs):
            # ----- FORWARD + LOSS (entrenamiento) -----
            y_predicho = self.modelo(self.x_entrenamiento)
            loss_train = criterion(y_predicho, self.y_entrenamiento)

            # ----- BACKWARD -----
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # ----- EVALUACIÓN DE VALIDACIÓN (sin gradientes) -----
            # [CAMBIO 2] Se calcula loss_val en cada época para el historial dual.
            with torch.no_grad():
                y_val_pred = self.modelo(self.x_validacion)
                loss_val   = criterion(y_val_pred, self.y_validacion)

            # Registrar ambas pérdidas
            self.historial_loss_train.append(loss_train.item())
            self.historial_loss_val.append(loss_val.item())

            # Reporte cada 200 épocas
            if (epoch + 1) % 200 == 0:
                print(f"  Época [{epoch + 1:5d}/{epochs}] | "
                      f"Loss Train: {loss_train.item():.6f} | "
                      f"Loss Val:   {loss_val.item():.6f}")

        print("✓ Entrenamiento completado.")
        print(f"  Loss Train final : {self.historial_loss_train[-1]:.8f}")
        print(f"  Loss Val   final : {self.historial_loss_val[-1]:.8f}")

    # --------------------------------------------------------------------------
    # PREDICCIÓN
    # --------------------------------------------------------------------------

    def predecir(self, x):
        """Genera predicciones para un array de entrada x."""
        if self.modelo is None:
            raise ValueError("Modelo no construido.")

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            predicciones = self.modelo(x)
        return predicciones

    # --------------------------------------------------------------------------
    # VISUALIZACIÓN CON CURVAS DE PÉRDIDA DUALES
    # --------------------------------------------------------------------------

    def graficar_resultados(self, nombre_archivo="prediccion_vs_real.png"):
        """
        Genera y guarda las gráficas de comparación y pérdida dual.

        [CAMBIO 2] La gráfica de pérdida ahora incluye dos curvas:
          - loss_train (azul): pérdida sobre datos de entrenamiento por época.
          - loss_val   (rojo): pérdida sobre datos de validación por época.

        La divergencia entre ambas curvas indica sobreajuste; la convergencia
        hacia valores cercanos a cero con ambas curvas paralelas indica un modelo
        bien generalizado.

        Subplots:
          1. Predicción de la red vs. sin(x) real.
          2. Curvas de pérdida dual en escala logarítmica.
        """
        if self.modelo is None:
            return

        # Datos de prueba ordenados para curva suave
        x_test       = np.linspace(0, 2 * np.pi, 200)
        y_test_real  = np.sin(x_test)
        y_test_pred  = self.predecir(x_test).numpy()

        # Recuperar información de la activación para el título
        activacion = self.modelo[1]
        titulo_activacion = (f"f(x) = {activacion.alpha_1:.4f}·x³ "
                             f"+ {activacion.alpha_2:.4f}·x")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Activación Hamiltoniana: {titulo_activacion}",
                     fontsize=13, fontweight='bold')

        # ------------------------------------------------------------------
        # Subplot 1: Predicción vs. sin(x) real
        # ------------------------------------------------------------------
        ax1 = axes[0]
        ax1.plot(x_test, y_test_real,
                 label="Real — sin(x)", color="steelblue", alpha=0.8, linewidth=2.5)
        ax1.plot(x_test, y_test_pred,
                 label="Red Neuronal", color="crimson", linestyle="--", linewidth=2.5)
        ax1.scatter(self.x_entrenamiento.numpy(),
                    self.y_entrenamiento.numpy(),
                    s=4, color="gray", alpha=0.25, label="Datos Entrenamiento")
        ax1.set_title("Ajuste de Función Senoidal")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ------------------------------------------------------------------
        # Subplot 2: Curvas de pérdida duales (loss_train y loss_val por época)
        # ------------------------------------------------------------------
        # [CAMBIO 2] Ambas curvas en escala logarítmica para observar convergencia.
        # La brecha entre ellas indica el grado de sobreajuste del modelo.
        ax2 = axes[1]
        epocas = range(1, len(self.historial_loss_train) + 1)

        ax2.plot(epocas, self.historial_loss_train,
                 label="Loss Train (MSE)", color="royalblue", linewidth=1.5, alpha=0.9)
        ax2.plot(epocas, self.historial_loss_val,
                 label="Loss Val (MSE)",   color="tomato",     linewidth=1.5, alpha=0.9,
                 linestyle="--")

        # Marcar el mínimo de loss_val con una anotación
        idx_min_val    = int(np.argmin(self.historial_loss_val))
        min_val_value  = self.historial_loss_val[idx_min_val]
        ax2.axvline(x=idx_min_val + 1, color='gray', linestyle=':', alpha=0.5)
        ax2.annotate(
            f"min val\n{min_val_value:.5f}",
            xy=(idx_min_val + 1, min_val_value),
            xytext=(idx_min_val + 1 + len(epocas) * 0.05, min_val_value * 2),
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='gray'),
            color='gray'
        )

        ax2.set_title("Evolución de la Pérdida (Train vs. Validación)")
        ax2.set_xlabel("Épocas")
        ax2.set_ylabel("Pérdida (MSE, escala log)")
        ax2.set_yscale("log")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(nombre_archivo, dpi=150)
        print(f"\n✓ Gráfica guardada como '{nombre_archivo}'")
        plt.show()

    # --------------------------------------------------------------------------
    # GUARDADO DEL MODELO
    # --------------------------------------------------------------------------

    def guardar_modelo(self, path="modelo_senoidal.pth"):
        """Guarda los parámetros del modelo entrenado en disco."""
        torch.save(self.modelo.state_dict(), path)
        print(f"✓ Modelo guardado en '{path}'")


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROYECTO PYTORCH 2026 — BÚSQUEDA DE PARÁMETROS ALPHA HAMILTONIANOS")
    print("Autor: Daniel Cano Duque")
    print("=" * 70)

    proy = ModeloSenoidal()

    # 1. Generar datos con mezcla aleatoria
    proy.generar_datos()

    # 2. Búsqueda en cuadrícula para encontrar los valores óptimos de alpha_1 y alpha_2.
    #    - n_valores=20 genera una cuadrícula de 20×20 = 400 combinaciones.
    #    - epochs_busqueda=500 es suficiente para identificar tendencias de convergencia
    #      sin el costo computacional del entrenamiento completo (2000 épocas).
    mejor_a1, mejor_a2, tabla_resultados = proy.buscar_parametros_optimos(
        n_valores=20,
        rango_alpha=(1.0, 5.0),
        epochs_busqueda=500,
        lr=0.01
    )

    # 3. Construir el modelo final con los parámetros óptimos encontrados
    proy.construir_modelo(alpha_1=mejor_a1, alpha_2=mejor_a2)

    # 4. Entrenar el modelo final con más épocas para convergencia completa.
    #    Se usan 2000 épocas (mismo valor que la versión anterior) para el modelo
    #    definitivo, ahora con los parámetros alpha óptimos.
    proy.entrenar(epochs=2000, lr=0.01)

    # 5. Graficar resultados con curvas de pérdida duales (train + val)
    proy.graficar_resultados()

    # 6. Guardar parámetros del modelo óptimo entrenado
    proy.guardar_modelo()