# ayudantes.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

def obtener_posicion_mas_frecuente(posiciones):
    # Contar la frecuencia de cada posición
    posiciones_lista = [pos.strip() for sublist in posiciones.str.split(',') for pos in sublist]
    # print(posiciones_lista)
    posicion_frecuente = pd.Series(posiciones_lista).mode()[0]  # Obtener la posición más frecuente
    return posicion_frecuente

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

# Diccionario de equivalencias
equivalencias_posiciones = {
    'GK': 'PT', 'DF': 'DF', 'MF': 'CC', 'FW': 'DL',
    'FB': 'LT', 'LB': 'LI', 'RB': 'LD', 'CB': 'CT',
    'DM': 'MCD', 'CM': 'MC', 'LM': 'II', 'RM': 'ID',
    'WM': 'INT', 'LW': 'EI', 'RW': 'ED', 'AM': 'MCO'
}

def mapear_posiciones(posiciones):
    # Separar las posiciones si hay más de una y mapear cada una usando el diccionario
    posiciones_mapeadas = [equivalencias_posiciones.get(pos.strip(), pos) for pos in posiciones.split(',')]
    # Unir de nuevo las posiciones mapeadas en una cadena
    return ','.join(posiciones_mapeadas)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

def generar_grafico_similitud(jugadores_similares, jugador):
    """
    Genera un gráfico de barras con las similitudes entre jugadores.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=jugadores_similares.index, y=jugadores_similares.values, ax=ax)
    
    ax.set_xlabel('Jugadores')
    ax.set_ylabel('Similitud')
    ax.set_title(f'Similitud con Jugador {jugador}')

    # Ajustar el ángulo de las etiquetas en el eje x
    plt.xticks(rotation=15, ha='right')  # Girar las etiquetas 45 grados para mejorar la legibilidad
    
    # Ajustar el tamaño de las etiquetas si es necesario
    ax.tick_params(axis='x', labelsize=6)  # Ajusta el tamaño de las etiquetas del eje x
    
    # Ajustar el layout para que las etiquetas no se corten
    plt.tight_layout()
    
    return fig

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

def generar_grafico_radar_1(jugador_data, jugador, posicion):

    columnas_a_radiar = jugador_data.columns

    # Número de variables
    num_variables = len(columnas_a_radiar)

    # Ángulos para el gráfico
    angulos = [n / float(num_variables) * 2 * pi for n in range(num_variables)]
    angulos += angulos[:1]  # Cerrar el círculo

    # Iniciar la figura
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Dibujar una línea en el gráfico
    valores = jugador_data.values.flatten().tolist()
    valores += valores[:1]  # Cerrar el círculo
    ax.plot(angulos, valores, linewidth=2, linestyle='solid')

    # Rellenar el gráfico
    ax.fill(angulos, valores, 'b', alpha=0.4)

    # Añadir etiquetas de cada una de las variables
    plt.xticks(angulos[:-1], columnas_a_radiar, color='black', size=8)

    # Añadir título
    plt.title(f'Gráfico de Radar para {jugador} en {posicion}', size=15, color='blue', y=1.1)

    # En lugar de plt.show(), devolvemos el objeto 'fig'
    return fig

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

def generar_grafico_radar_2(jugador1_data, jugador2_data, jugador1, jugador2, posicion1, posicion2):
    
    columnas_a_radiar = jugador1_data.columns
    
    # Número de variables
    num_variables = len(columnas_a_radiar)

    # Ángulos para el gráfico
    angulos = [n / float(num_variables) * 2 * pi for n in range(num_variables)]
    angulos += angulos[:1]  # Cerrar el círculo

    # Iniciar la figura
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Dibujar las líneas para cada jugador
    for jugador_data, jugador_nombre in zip([jugador1_data, jugador2_data], [jugador1, jugador2]):
        valores = jugador_data.values.flatten().tolist()
        valores += valores[:1]  # Cerrar el círculo
        ax.plot(angulos, valores, linewidth=2, linestyle='solid', label=jugador_nombre)
        ax.fill(angulos, valores, alpha=0.4)  # Rellenar el área bajo la línea

    # Añadir etiquetas de cada una de las variables
    plt.xticks(angulos[:-1], columnas_a_radiar, color='black', size=8)

    # Añadir título y leyenda
    plt.title(f'Gráfico de Radar para {jugador1} ({posicion1}) y {jugador2} ({posicion2})', size=15, color='blue', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Devolver el objeto 'fig'
    return fig