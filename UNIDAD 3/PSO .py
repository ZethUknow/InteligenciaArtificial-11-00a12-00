import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pyswarms as ps

#1. CONFIGURACIÓN DEL ENTORNO SIMULADO ---

# Área de estudio (ej. un campo de 100m x 100m)
TAMANO_CAMPO = 100

# Número de sensores que queremos optimizar
N_SENSORES = 5

# Número de partículas en el enjambre (cuántas soluciones probamos a la vez)
N_PARTICULAS = 50

# Dimensiones del problema: cada sensor tiene (x, y), por lo que = N_SENSORES * 2
N_DIMENSIONES = N_SENSORES * 2


#2. SIMULACIÓN DE VARIABLES (Topografía, Cultivo, Suelo) ---

# Creamos un "Mapa de Variabilidad" sintético para Guasave.
# Usaremos 3 "puntos calientes" (Gaussianas) para simular:
# 1. Una zona de tomates (alta necesidad de agua)
# 2. Una zona de baja elevación (riesgo de salinidad/encharcamiento)
# 3. Una zona con baja materia orgánica
# Estos son los datos "aleatorios/básicos" que solicitaste, pero con estructura.

# Coordenadas [x, y] de los centros de los hotspots
coords_hotspot_1 = [25, 30]  # Zona de "Tomates"
coords_hotspot_2 = [70, 70]  # Zona de "Salinidad"
coords_hotspot_3 = [30, 80]  # Zona de "Baja Materia Orgánica"

# Creamos las distribuciones normales (campanas de Gauss)
variabilidad_1 = multivariate_normal(mean=coords_hotspot_1, cov=[[30, 0], [0, 30]])
variabilidad_2 = multivariate_normal(mean=coords_hotspot_2, cov=[[50, 0], [0, 50]])
variabilidad_3 = multivariate_normal(mean=coords_hotspot_3, cov=[[20, 0], [0, 20]])

def obtener_variabilidad(x, y):
    """
    Calcula la "necesidad de monitoreo" combinada en un punto (x, y) del campo.
    Un valor más alto significa que es más importante tener un sensor cerca.
    """
    # Apilamos X e Y en la última dimensión para crear un grid de puntos (x,y)
    # Forma de entrada x: (ej. 100, 100), y: (ej. 100, 100)
    # Forma de salida pos: (ej. 100, 100, 2)
    pos_apiladas = np.stack([x, y], axis=-1)
    
    # Sumamos la influencia de todos los hotspots
    return (variabilidad_1.pdf(pos_apiladas) * 500 + 
            variabilidad_2.pdf(pos_apiladas) * 700 + 
            variabilidad_3.pdf(pos_apiladas) * 400)

#3. DISEÑO DE LA FUNCIÓN DE COSTO (FITNESS FUNCTION) ---

def funcion_costo(lote_particulas):
    """
    Esta es la función que PSO intentará MINIMIZAR.
    Toma un 'lote' de partículas (un array de numpy).
    Cada fila 'particula' en 'lote_particulas' es una solución completa.
    """
    
    # 'lote_particulas' tiene forma (N_PARTICULAS, N_DIMENSIONES)
    # Ejemplo: (50, 10) si hay 50 partículas y 5 sensores (10 dims)
    
    costos = []
    
    # Creamos una grilla de 20x20 para muestrear el campo
    # (Evaluar en cada cm sería muy lento, esto es una buena aproximación)
    puntos_grilla = np.linspace(0, TAMANO_CAMPO, 20)
    
    # Iteramos sobre cada partícula (cada solución candidata)
    for particula in lote_particulas:
        # particula es un vector 1D de 10 elementos: [x1, y1, x2, y2, ...]
        # Lo convertimos a una forma útil: (N_SENSORES, 2)
        # Ejemplo: [[x1, y1], [x2, y2], ...]
        posiciones_sensores = particula.reshape(N_SENSORES, 2)
        
        costo_total_particula = 0
        
        # Iteramos sobre cada punto (gx, gy) de nuestra grilla del campo
        for gx in puntos_grilla:
            for gy in puntos_grilla:
                # 1. Obtener la variabilidad (importancia) de este punto
                variabilidad = obtener_variabilidad(gx, gy)
                
                # 2. Calcular la distancia de este punto (gx, gy) a TODOS los sensores
                # Usamos la norma de NumPy (distancia euclidiana)
                punto = np.array([gx, gy])
                distancias = np.linalg.norm(posiciones_sensores - punto, axis=1)
                
                # 3. Encontrar la distancia al sensor MÁS CERCANO
                distancia_minima = np.min(distancias)
                
                # 4. Calcular el costo para este punto
                # "El costo es alto si la variabilidad es alta Y el sensor más cercano está lejos"
                costo_punto = variabilidad * distancia_minima
                
                costo_total_particula += costo_punto
                
        # Agregamos el costo total de esta partícula a la lista
        costos.append(costo_total_particula)
        
    # Devolvemos un array 1D con el costo de cada partícula
    return np.array(costos)

#4. CONFIGURACIÓN Y EJECUCIÓN DE PSO

# Opciones de PSO (parámetros cognitivo 'c1', social 'c2', e inercia 'w')
# Estos valores son estándares y funcionan bien
opciones = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Límites del espacio de búsqueda
# Cada coordenada (x o y) debe estar dentro del campo (0 a TAMANO_CAMPO)

# Límite inferior (un array de N_DIMENSIONES ceros)
limites_minimos = np.zeros(N_DIMENSIONES)
# Límite superior (un array de N_DIMENSIONES "100")
limites_maximos = np.ones(N_DIMENSIONES) * TAMANO_CAMPO

# Creamos una tupla de límites
limites = (limites_minimos, limites_maximos)

# Instanciamos el optimizador (GlobalBestPSO es el más común)
optimizador = ps.single.GlobalBestPSO(n_particles=N_PARTICULAS,
                                    dimensions=N_DIMENSIONES,
                                    options=opciones,
                                    bounds=limites)

print(" INICIANDO OPTIMIZACIÓN PSO...")
print(f"Buscando posiciones óptimas para {N_SENSORES} sensores.")
print(f"Número de partículas (soluciones) por iteración: {N_PARTICULAS}")
print(f"Dimensiones del problema: {N_DIMENSIONES}")
print("---")

# Ejecutamos la optimización
# 'iters=100' significa que el enjambre "volará" por 100 iteraciones
# 'verbose=3' imprimirá el progreso en cada iteración (como solicitaste)
costo_final, posicion_optima = optimizador.optimize(funcion_costo, iters=100, verbose=3)

print("---")
print(" ¡OPTIMIZACIÓN COMPLETADA!")
print(f"Costo (Fitness) mínimo encontrado: {costo_final:.4f}")

# 'posicion_optima' contiene las coordenadas óptimas
posiciones_optimas = posicion_optima.reshape(N_SENSORES, 2)
print("Las coordenadas óptimas para los sensores son:")
for i, (x, y) in enumerate(posiciones_optimas):
    print(f"  Sensor {i+1}: (x={x:.2f}, y={y:.2f})")


#5. VISUALIZACIÓN DE RESULTADOS 

print("\n Generando visualización de resultados...")

# Crear una malla para graficar el mapa de variabilidad
x_plot = np.linspace(0, TAMANO_CAMPO, 100)
y_plot = np.linspace(0, TAMANO_CAMPO, 100)
X, Y = np.meshgrid(x_plot, y_plot)
Z = obtener_variabilidad(X, Y)

plt.figure(figsize=(12, 9))

# Dibujar el mapa de variabilidad (contorno)
plt.contourf(X, Y, Z, levels=20, cmap='YlGn')
plt.colorbar(label='Índice de Variabilidad (Suelo/Cultivo/Topo)')

# Dibujar las posiciones óptimas de los sensores
plt.scatter(posiciones_optimas[:, 0], posiciones_optimas[:, 1], 
            c='red', s=150, marker='X', label='Posición Óptima del Sensor')

# Dibujar los "hotspots" originales para referencia
plt.scatter([coords_hotspot_1[0], coords_hotspot_2[0], coords_hotspot_3[0]], 
            [coords_hotspot_1[1], coords_hotspot_2[1], coords_hotspot_3[1]], 
            c='blue', s=50, marker='*', label='Centros de Variabilidad (Simulados)')

plt.title(f'Optimización PSO para Ubicación de {N_SENSORES} Sensores en Guasave (Simulada)', fontsize=16)
plt.xlabel('Coordenada X del Campo (m)')
plt.ylabel('Coordenada Y del Campo (m)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.axis('equal')
plt.show()