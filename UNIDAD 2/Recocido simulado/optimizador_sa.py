import numpy as np
import random

class RecocidoSimulado:
    """
    Clase que implementa el algoritmo de Recocido Simulado para resolver el VRP.
    """
    def __init__(self, matriz_costos, temp_inicial, temp_final, tasa_enfriamiento):
        """
        Inicializa el optimizador.

        Args:
            matriz_costos (np.array): Matriz de costos entre nodos.
            temp_inicial (float): Temperatura inicial del sistema.
            temp_final (float): Temperatura final (criterio de parada).
            tasa_enfriamiento (float): Tasa con la que disminuye la temperatura (e.g., 0.99).
        """
        self.matriz_costos = matriz_costos
        self.temp_inicial = temp_inicial
        self.temp_final = temp_final
        self.tasa_enfriamiento = tasa_enfriamiento
        self.historial_costos = []

    def calcular_costo_ruta(self, ruta):
        """Calcula el costo total de una ruta."""
        costo_total = 0
        for i in range(len(ruta) - 1):
            costo_total += self.matriz_costos[ruta[i], ruta[i+1]]
        return costo_total

    def generar_vecino(self, ruta):
        """
        Genera una solución vecina aplicando un intercambio 2-opt.
        Esto evita crear rutas inválidas y es un método estándar para el VRP.
        """
        vecino = ruta[:]
        # Seleccionar dos índices distintos, excluyendo el primero y el último (CEDIS)
        i, j = random.sample(range(1, len(vecino) - 1), 2)
        if i > j:
            i, j = j, i
        # Invertir el segmento de la ruta entre i y j
        vecino[i:j+1] = reversed(vecino[i:j+1])
        return vecino

    def optimizar(self, ruta_inicial):
        """
        Ejecuta el algoritmo de Recocido Simulado.

        Args:
            ruta_inicial (list): La primera ruta a evaluar.

        Returns:
            tuple: La mejor ruta encontrada y su costo.
        """
        temp_actual = self.temp_inicial
        solucion_actual = ruta_inicial
        costo_actual = self.calcular_costo_ruta(solucion_actual)
        
        mejor_solucion = solucion_actual
        mejor_costo = costo_actual
        
        self.historial_costos.append(costo_actual)

        while temp_actual > self.temp_final:
            # Generar una solución vecina
            vecino = self.generar_vecino(solucion_actual)
            costo_vecino = self.calcular_costo_ruta(vecino)
            
            # Calcular diferencia de costo
            delta_costo = costo_vecino - costo_actual

            # Decidir si se acepta la nueva solución
            if delta_costo < 0 or random.uniform(0, 1) < np.exp(-delta_costo / temp_actual):
                solucion_actual = vecino
                costo_actual = costo_vecino
            
            # Actualizar la mejor solución encontrada hasta ahora
            if costo_actual < mejor_costo:
                mejor_solucion = solucion_actual
                mejor_costo = costo_actual
            
            self.historial_costos.append(mejor_costo)
            
            # Enfriar el sistema
            temp_actual *= self.tasa_enfriamiento
            
        return mejor_solucion, mejor_costo