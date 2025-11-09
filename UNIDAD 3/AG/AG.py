"""
Algoritmo Genético para resolver el Problema del Viajante (TSP).
Este script busca la ruta más corta entre varios 'municipios' (puntos).
"""

import operator
import random
from typing import List, Tuple

import numpy as np
import pandas as pd


class Municipio:
    """
    Representa un 'municipio' (un punto) con coordenadas X e Y.
    """

    def __init__(self, x: float, y: float):
        """Guarda las coordenadas X e Y del municipio."""
        self.x = x
        self.y = y

    def distancia(self, otro_municipio: 'Municipio') -> float:
        """Calcula la distancia (en línea recta) a otro municipio."""
        x_dis = abs(self.x - otro_municipio.x)
        y_dis = abs(self.y - otro_municipio.y)
        distancia = np.sqrt((x_dis ** 2) + (y_dis ** 2))
        return distancia

    def __repr__(self) -> str:
        """Define cómo se 'imprime' un municipio (ej: "(40.4, -3.7)")."""
        return f"({self.x}, {self.y})"


class Ruta:
    """
    Representa una 'ruta' (una solución).
    Es una lista de municipios en un orden específico.
    """

    def __init__(self, ruta: List[Municipio]):
        """Guarda la lista de municipios que forman la ruta."""
        self.ruta = ruta
        self.distancia = 0.0
        self.aptitud = 0.0

    def calcular_distancia(self) -> float:
        """
        Suma la distancia total de la ruta (incluyendo la vuelta al inicio).
        Solo calcula la primera vez; después, usa el valor guardado.
        """
        if self.distancia == 0:
            distancia_relativa = 0.0
            for i in range(len(self.ruta)):
                punto_inicial = self.ruta[i]

                # Si es el último punto, debe conectarse con el primero
                if i + 1 < len(self.ruta):
                    punto_final = self.ruta[i + 1]
                else:
                    punto_final = self.ruta[0]

                distancia_relativa += punto_inicial.distancia(punto_final)
            self.distancia = distancia_relativa
        return self.distancia

    def calcular_aptitud(self) -> float:
        """
        Calcula la 'aptitud' de la ruta. Es (1 / distancia).
        Rutas más cortas tienen mayor aptitud (son 'mejores').
        """
        if self.aptitud == 0:
            # Llama a calcular_distancia() para asegurarse de que exista
            self.aptitud = 1.0 / float(self.calcular_distancia())
        return self.aptitud


class AlgoritmoGeneticoTSP:
    """
    Controla todo el proceso del algoritmo genético:
    Crea la población, la evalúa y la evoluciona generación tras generación.
    """

    def __init__(self,
                 municipios: List[Municipio],
                 tamano_poblacion: int,
                 tamano_elite: int,
                 tasa_mutacion: float):
        """
        Prepara el algoritmo con los parámetros iniciales.

        Args:
            municipios: La lista de todos los puntos a visitar.
            tamano_poblacion: Cuántas rutas (soluciones) hay por generación.
            tamano_elite: Cuántas de las MEJORES rutas pasan a la siguiente
                          generación sin cambios.
            tasa_mutacion: Qué tan probable es (0.0 a 1.0) que una ruta cambie
                           aleatoriamente.
        """
        self.municipios = municipios
        self.tamano_poblacion = tamano_poblacion
        self.tamano_elite = tamano_elite
        self.tasa_mutacion = tasa_mutacion
        
        # Inicia el algoritmo creando la primera población
        self.poblacion = self._crear_poblacion_inicial()

    # --- 1. Métodos Principales (Cómo se usa) ---

    def ejecutar(self, num_generaciones: int) -> Ruta:
        """
        Inicia el algoritmo y lo corre por N generaciones.
        Este es el método principal que se llama desde fuera.
        """
        # Imprimir estado inicial
        mejor_ruta_inicial = self.obtener_mejor_ruta_actual()
        print(f"Distancia Inicial (Generación 0): {mejor_ruta_inicial.calcular_distancia():.2f}")

        # El ciclo evolutivo
        for i in range(num_generaciones):
            self._evolucionar_generacion()

            # Imprimir progreso cada 50 generaciones
            if (i + 1) % 50 == 0:
                mejor_ruta_actual = self.obtener_mejor_ruta_actual()
                # Usamos .distancia porque ya fue calculada en obtener_mejor_ruta
                print(f"Generación {i+1:4} | Mejor Distancia: {mejor_ruta_actual.distancia:.2f}")

        # Imprimir estado final
        mejor_ruta_final = self.obtener_mejor_ruta_actual()
        print(f"\nDistancia Final (Generación {num_generaciones}): {mejor_ruta_final.calcular_distancia():.2f}")
        return mejor_ruta_final

    # --- 2. El Motor de Evolución ---

    def _evolucionar_generacion(self):
        """
        Es el 'motor' del algoritmo. Realiza un ciclo completo de evolución:
        1. Clasifica la población actual.
        2. Selecciona a los mejores (padres).
        3. Crea hijos (cruce).
        4. Aplica cambios aleatorios (mutación).
        5. Reemplaza la población antigua por la nueva.
        """
        # 1. Evaluar y clasificar la población actual
        pop_clasificada = self._clasificar_poblacion(self.poblacion)

        # 2. Decidir quién se reproduce (devuelve índices)
        indices_seleccionados = self._seleccion(pop_clasificada)

        # 3. Obtener los objetos Ruta de los seleccionados
        grupo_apa = self._grupo_apareamiento(indices_seleccionados)

        # 4. Crear la nueva generación (Cruce + Elitismo)
        poblacion_hijos = self._crear_generacion_cruzada(grupo_apa)

        # 5. Aplicar mutaciones aleatorias (se salta la élite)
        nueva_generacion = self._mutar_poblacion(poblacion_hijos)

        # 6. Actualizar la población para el siguiente ciclo
        self.poblacion = nueva_generacion

    def obtener_mejor_ruta_actual(self) -> Ruta:
        """Revisa la población actual y devuelve la mejor ruta (la más corta)."""
        # Clasifica la población y toma el primero (índice 0)
        pop_clasificada = self._clasificar_poblacion(self.poblacion)
        mejor_indice = pop_clasificada[0][0]
        return self.poblacion[mejor_indice]

    # --- 3. Pasos de la Evolución (Helpers) ---

    # --- PASO A: INICIALIZACIÓN ---

    def _crear_poblacion_inicial(self) -> List[Ruta]:
        """Crea la primera 'generación' de rutas. Todas son aleatorias."""
        return [self._crear_ruta_aleatoria() for _ in range(self.tamano_poblacion)]

    def _crear_ruta_aleatoria(self) -> Ruta:
        """Crea una única ruta desordenando la lista de municipios."""
        ruta_lista = random.sample(self.municipios, len(self.municipios))
        return Ruta(ruta_lista)

    # --- PASO B: EVALUACIÓN Y SELECCIÓN ---

    def _clasificar_poblacion(self, poblacion: List[Ruta]) -> List[Tuple[int, float]]:
        """
        Calcula la 'aptitud' de cada ruta en la población
        y las ordena de mejor (más apta) a peor (menos apta).
        """
        resultados_aptitud = {}
        for i, ruta_obj in enumerate(poblacion):
            # Calcula la aptitud (1 / distancia)
            resultados_aptitud[i] = ruta_obj.calcular_aptitud()

        # Devuelve una lista ordenada: [(indice_ruta_5, aptitud_alta), (indice_ruta_2, aptitud_media), ...]
        return sorted(resultados_aptitud.items(), key=operator.itemgetter(1), reverse=True)

    def _seleccion(self, pop_clasificada: List[Tuple[int, float]]) -> List[int]:
        """
        Decide qué rutas 'sobreviven' para crear la siguiente generación.
        Combina dos métodos:
        1. 'Elitismo': Los N mejores (tamano_elite) pasan automáticamente.
        2. 'Ruleta': El resto se elige al azar, pero dando más
           probabilidad a las rutas que tienen mejor aptitud.
        """
        indices_seleccionados = []

        # 1. Elitismo: Añadir los mejores N
        for i in range(self.tamano_elite):
            indices_seleccionados.append(pop_clasificada[i][0])

        # 2. Selección por Ruleta (para el resto)
        # Prepara los datos para la ruleta
        df = pd.DataFrame(np.array(pop_clasificada), columns=["Indice", "Aptitud"])
        df['cum_sum'] = df.Aptitud.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Aptitud.sum()

        # Seleccionar los (N - elite) individuos restantes
        for _ in range(self.tamano_poblacion - self.tamano_elite):
            seleccion = 100 * random.random() # Elegir un número al azar (0-100)
            for i in range(len(pop_clasificada)):
                # Detenerse en cuanto el porcentaje acumulado supera al azar
                if seleccion <= df.iat[i, 3]: # iat[i, 3] es 'cum_perc'
                    indices_seleccionados.append(pop_clasificada[i][0])
                    break
        
        return indices_seleccionados

    def _grupo_apareamiento(self, indices_seleccionados: List[int]) -> List[Ruta]:
        """
        Junta a los individuos seleccionados (por sus índices)
        en un 'grupo de padres' para la reproducción.
        """
        return [self.poblacion[i] for i in indices_seleccionados]

    # --- PASO C: REPRODUCCIÓN (CRUCE) ---

    def _crear_generacion_cruzada(self, grupo_apa: List[Ruta]) -> List[Ruta]:
        """
        Crea la nueva generación (hijos).
        1. Mantiene a la 'élite' (los N mejores) intacta.
        2. Crea 'hijos' cruzando a los padres del resto del grupo.
        """
        hijos = []

        # 1. Preservar la élite
        for i in range(self.tamano_elite):
            hijos.append(grupo_apa[i])

        # 2. Cruzar el resto para llenar la población
        # Barajamos el pool para que los cruces sean más aleatorios
        pool_cruce = random.sample(grupo_apa, len(grupo_apa))

        num_hijos_restantes = self.tamano_poblacion - self.tamano_elite

        for i in range(num_hijos_restantes):
            # Cruzar un individuo (i) con otro (len - i - 1)
            hijo = self._cruce(pool_cruce[i], pool_cruce[len(grupo_apa) - i - 1])
            hijos.append(hijo)

        return hijos

    def _cruce(self, padre1: Ruta, padre2: Ruta) -> Ruta:
        """
        Crea un 'hijo' a partir de dos 'padres' (Ordered Crossover).
        1. Toma un trozo aleatorio de la ruta del padre 1.
        2. Rellena los huecos con los municipios del padre 2, en orden,
           y sin repetir los que ya se tomaron del padre 1.
        """
        hijo_genes = []
        genes_p1 = []
        
        # 1. Seleccionar un "trozo" aleatorio de la ruta del padre 1
        gen_a = int(random.random() * len(padre1.ruta))
        gen_b = int(random.random() * len(padre1.ruta))
        gen_inicio = min(gen_a, gen_b)
        gen_fin = max(gen_a, gen_b)

        # Copiar ese trozo al hijo
        for i in range(gen_inicio, gen_fin):
            genes_p1.append(padre1.ruta[i])

        # 2. Tomar los genes del padre 2 que NO estén ya en el hijo
        genes_p2 = [item for item in padre2.ruta if item not in genes_p1]

        # 3. Combinar: el trozo del padre 1 + lo restante del padre 2
        hijo_genes = genes_p1 + genes_p2

        return Ruta(hijo_genes)

    # --- PASO D: MUTACIÓN ---

    def _mutar_poblacion(self, poblacion_hijos: List[Ruta]) -> List[Ruta]:
        """
        Aplica la mutación a toda la nueva generación.
        IMPORTANTE: Se salta a la 'élite' (los primeros N),
        que se protegen para no empeorar.
        """
        poblacion_mutada = []

        # 1. Añadir la élite sin mutar
        for i in range(self.tamano_elite):
            poblacion_mutada.append(poblacion_hijos[i])

        # 2. Mutar el resto de la población
        for i in range(self.tamano_elite, len(poblacion_hijos)):
            individuo_mutado = self._mutar_individuo(poblacion_hijos[i])
            poblacion_mutada.append(individuo_mutado)

        return poblacion_mutada

    def _mutar_individuo(self, individuo: Ruta) -> Ruta:
        """
        Cambia aleatoriamente dos municipios de lugar dentro de una misma ruta.
        Esto se hace según la 'tasa_mutacion' y sirve para
        introducir variedad y evitar que el algoritmo se atasque.
        """
        ruta_mutada_lista = list(individuo.ruta) # Trabajar sobre una copia

        for indice_intercambio in range(len(ruta_mutada_lista)):
            # Si el "dado" (random) es menor que la tasa, ocurre la mutación
            if random.random() < self.tasa_mutacion:
                # Encontrar un segundo índice aleatorio para intercambiar
                indice_con = int(random.random() * len(ruta_mutada_lista))

                # Intercambiar (Swap)
                municipio1 = ruta_mutada_lista[indice_intercambio]
                municipio2 = ruta_mutada_lista[indice_con]
                ruta_mutada_lista[indice_intercambio] = municipio2
                ruta_mutada_lista[indice_con] = municipio1

        # Devuelve una NUEVA ruta con los cambios
        return Ruta(ruta_mutada_lista)


# --- Bloque de Ejecución Principal ---

if __name__ == "__main__":

    # 1. Definir la lista de municipios (ciudades/puntos)
    lista_de_municipios = [
        Municipio(x=40.4168, y=-3.7038),  # Madrid
        Municipio(x=41.3851, y=2.1734),   # Barcelona
        Municipio(x=39.4699, y=-0.3763),  # Valencia
        Municipio(x=37.3891, y=-5.9845),  # Sevilla
        Municipio(x=41.6488, y=-0.8891),  # Zaragoza
        Municipio(x=36.7213, y=-4.4214),  # Málaga
        Municipio(x=37.9922, y=-1.1307),  # Murcia
        Municipio(x=43.2630, y=-2.9350),  # Bilbao
        Municipio(x=43.3623, y=-8.4115),  # A Coruña
        Municipio(x=39.5696, y=2.6502)    # Palma
    ]

    # 2. Configurar e instanciar el Algoritmo Genético
    ag_tsp = AlgoritmoGeneticoTSP(
        municipios=lista_de_municipios,
        tamano_poblacion=100, # 100 rutas por generación
        tamano_elite=20,       # Las 20 mejores rutas se salvan siempre
        tasa_mutacion=0.01     # 1% de probabilidad de que un municipio cambie
    )

    # 3. Ejecutar el algoritmo
    mejor_ruta = ag_tsp.ejecutar(num_generaciones=500)

    print("\nMejor ruta encontrada:")
    # Imprime la secuencia de coordenadas de la mejor ruta
    print(mejor_ruta.ruta)