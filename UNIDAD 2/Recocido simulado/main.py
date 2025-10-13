import random
import numpy as np
import matplotlib.pyplot as plt
from utils import cargar_datos, plot_ruta, plot_convergencia
from optimizador_sa import RecocidoSimulado

def main():
    print("🚀 Iniciando Proceso de Optimización de Rutas...")

    # --- 1. Carga de Datos ---
    RUTA_TIENDAS = 'data/datos_distribucion_tiendas.xlsx - Sheet1.csv'
    RUTA_COSTOS = 'data/matriz_costos_combustible.xlsx - Sheet1.csv'
    
    df_tiendas, matriz_costos = cargar_datos(RUTA_TIENDAS, RUTA_COSTOS)
    if df_tiendas is None:
        return # Termina si los datos no se pudieron cargar

    # --- 2. Definición del Escenario de Prueba ---
    # Para este ejemplo, usaremos el 'Centro de Distribución 2' y todas las tiendas de 'Nivel A'
    
    # Obtenemos el índice del CEDIS de interés (índice basado en la posición en el CSV)
    CEDIS_NOMBRE = 'Centro de Distribución 2'
    idx_cedis = df_tiendas.index[df_tiendas['Nombre'] == CEDIS_NOMBRE].tolist()[0]
    
    # Obtenemos los índices de las tiendas objetivo
    tiendas_nivel_a = df_tiendas[df_tiendas['Nivel_Tienda'] == 'A']
    idx_tiendas_objetivo = tiendas_nivel_a.index.tolist()
    
    print(f"\n🚚 Escenario: Desde '{CEDIS_NOMBRE}' (Nodo {idx_cedis}) a {len(idx_tiendas_objetivo)} tiendas de Nivel A.")

    # Nodos que formarán parte de nuestra ruta
    nodos_problema_idx = [idx_cedis] + idx_tiendas_objetivo
    df_nodos_problema = df_tiendas.iloc[nodos_problema_idx]

    # --- 3. Crear una Ruta Inicial Aleatoria ---
    # La ruta debe empezar y terminar en el CEDIS
    ruta_inicial = [idx_cedis] + random.sample(idx_tiendas_objetivo, len(idx_tiendas_objetivo)) + [idx_cedis]

    # --- 4. Configurar y Ejecutar el Optimizador ---
    # Parámetros del Recocido Simulado (puedes experimentar con estos valores)
    TEMP_INICIAL = 10000
    TEMP_FINAL = 0.1
    TASA_ENFRIAMIENTO = 0.995 # Un enfriamiento más lento suele dar mejores resultados

    optimizador = RecocidoSimulado(matriz_costos, TEMP_INICIAL, TEMP_FINAL, TASA_ENFRIAMIENTO)
    
    costo_inicial = optimizador.calcular_costo_ruta(ruta_inicial)
    
    print("\njecutando Recocido Simulado... (esto puede tardar unos segundos)")
    ruta_optima, costo_optimo = optimizador.optimizar(ruta_inicial)
    print("Optimización completada.")

    # --- 5. Mostrar Resultados ---
    print("\n--- Resultados de la Optimización ---")
    print(f"Ruta Inicial: {ruta_inicial}")
    print(f"Costo Inicial: {costo_inicial:.2f} unidades monetarias")
    print("-" * 35)
    print(f"Mejor Ruta Encontrada: {ruta_optima}")
    print(f"Costo Óptimo: {costo_optimo:.2f} unidades monetarias")
    print(f"Mejora: {((costo_inicial - costo_optimo) / costo_inicial) * 100:.2f}%")
    
    # --- 6. Visualización ---
    # Gráfico de rutas
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Comparación de Rutas de Distribución', fontsize=16)
    
    plot_ruta(df_tiendas, ruta_inicial, f'Ruta Inicial Aleatoria (Costo: {costo_inicial:.2f})', axes[0])
    plot_ruta(df_tiendas, ruta_optima, f'Ruta Optimizada (Costo: {costo_optimo:.2f})', axes[1])
    
    plt.show()

    # Gráfico de convergencia del costo
    plot_convergencia(optimizador.historial_costos)


if __name__ == '__main__':
    main()