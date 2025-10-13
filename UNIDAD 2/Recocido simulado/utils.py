import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt

def cargar_datos(ruta_datos, ruta_costos):
    """
    Carga los datos de las tiendas y la matriz de costos desde archivos CSV.

    Args:
        ruta_datos (str): Ruta al archivo CSV con información de las tiendas.
        ruta_costos (str): Ruta al archivo CSV con la matriz de costos de combustible.

    Returns:
        tuple: Un DataFrame con los datos de las tiendas y una matriz de costos (np.array).
    """
    try:
        # --- CORRECCIÓN 1: Usar los parámetros de la función ---
        # Cargar la información de ubicaciones desde la ruta proporcionada.
        df_tiendas = pd.read_csv(ruta_datos)
        
        # Cargar la matriz de costos de combustible desde la ruta proporcionada.
        matriz_costos = pd.read_csv(ruta_costos, header=None).values
        
        print("Datos cargados correctamente.")
        print(f"Número total de nodos (CEDIS + Tiendas): {len(df_tiendas)}")
        print(f"Dimensiones de la matriz de costos: {matriz_costos.shape}")
        
        return df_tiendas, matriz_costos
        
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado. Verifica que la ruta y el nombre del archivo sean correctos.")
        print(f"Detalle del error: {e}")
        return None, None
    except Exception as e:
        print(f" Ocurrió un error inesperado al leer los archivos: {e}")
        return None, None

def plot_ruta(df_nodos, ruta, titulo, ax):
    """
    Dibuja una ruta en un objeto de ejes de Matplotlib.

    Args:
    df_nodos (DataFrame): DataFrame con la latitud y longitud de los nodos.
    ruta (list): Lista de índices de nodos que forman la ruta.
    titulo (str): Título para el gráfico.
    ax (matplotlib.axes.Axes): Ejes donde se dibujará el gráfico.
    """
    # Coordenadas de la ruta en orden
    coords = df_nodos.iloc[ruta][['Longitud_WGS84', 'Latitud_WGS84']].values
    
    # Dibujar las conexiones de la ruta
    ax.plot(coords[:, 0], coords[:, 1], 'o-', markersize=5, label='Ruta')
    
    # Marcar el CEDIS (punto de inicio/fin)
    cedis_coords = df_nodos.iloc[ruta[0]][['Longitud_WGS84', 'Latitud_WGS84']]
    ax.plot(cedis_coords[0], cedis_coords[1], 's', color='red', markersize=10, label='CEDIS')
    
    ax.set_title(titulo)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True)
    ax.legend()

def plot_convergencia(historial_costos):
    """
    Dibuja la evolución del costo a lo largo de las iteraciones.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(historial_costos, color='darkorange')
    plt.title('Evolución del Costo en Recocido Simulado')
    plt.xlabel('Iteración')
    plt.ylabel('Costo de Combustible (Unidades Monetarias)')
    plt.grid(True)
    plt.show()