# logica.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from ayudantes import *

#_________________________________________________________________
from sqlalchemy import engine
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
#_________________________________________________________________

# Carga de datos
def cargar_datos(ruta_csv):
    """
    Carga el archivo CSV con los datos de los jugadores y devuelve un DataFrame.
    """
    df = pd.read_csv(ruta_csv)
    
    # Renombrar columnas para facilitar el acceso
    df.columns = [
        'jugador', 'equipo', 'numero', 'nacionalidad', 'posicion', 
        'edad', 'minutos', 'goles', 'asistencias', 'penaltis_marcados', 'penaltis_intentados',
        'total_tiros', 'tiros_a_puerta', 'amarillas', 'rojas', 'contactos', 'tackles', 'interceptaciones',
        'bloqueos', 'goles_esp', 'goles_esp_np', 'goles_esp_asistidos', 'acciones_creacion_tiro', 'acciones_creacion_gol', 
        'pases_completados', 'pases_intentados', 'pases_completados_p', 'pases_progresivos', 
        'conducciones', 'conducciones_progresivas', 'intentos_regate', 'regates_exitosos', 'fecha'
    ]
    
    # Retornar el DataFrame cargado
    return df

#_________________________________________________________________

import kagglehub
import pathlib

# Ruta de los archivos CSV (descargados automáticamente de Kaggle)
liga_csv = pathlib.Path(kagglehub.dataset_download("eduardopalmieri/laliga-players-stats")) / "database.csv"
premier_csv = pathlib.Path(kagglehub.dataset_download("eduardopalmieri/premier-league-player-stats-season-2425")) / "database.csv"
seriea_csv = pathlib.Path(kagglehub.dataset_download("eduardopalmieri/seria-a-players-stats-season-2425")) / "database.csv"

# Ruta de los archivos CSV (descargados manualmente de Kaggle)
# liga_csv = 'laliga/laliga-12.csv'
# premier_csv = 'laliga/premier-10.csv'
# seriea_csv = 'laliga/seriea-11.csv'

# Cargar los datos (este es un ejemplo, el CSV debe estar en la ruta indicada)
df_laliga = cargar_datos(liga_csv)
df_premier = cargar_datos(premier_csv)
df_seriea = cargar_datos(seriea_csv)

# Concatenar los DataFrames

df = pd.concat([df_laliga, df_premier, df_seriea], ignore_index=True)

#_________________________________________________________________

def generar_df_agrupado (df):

    columnas_sin_operar = ['jugador', 'equipo', 'numero', 'nacionalidad', 'posicion', 'edad']

    columnas_a_sumar = [
        'minutos', 'goles', 'asistencias', 'penaltis_marcados', 'penaltis_intentados', 
        'total_tiros', 'tiros_a_puerta', 'amarillas', 'rojas', 'contactos', 'tackles', 
        'interceptaciones', 'bloqueos', 'goles_esp', 'goles_esp_np', 'goles_esp_asistidos', 
        'acciones_creacion_tiro', 'acciones_creacion_gol', 'pases_completados', 'pases_intentados',
        'pases_progresivos', 'conducciones', 'conducciones_progresivas', 'intentos_regate', 'regates_exitosos'
    ]

    columnas_a_mediar = ['pases_completados_p']

    # Nos aseguramos que todos los valores estén en formato de texto antes de convertirlos a float
    # Convertimos la columna 'pases_completados_p' a 'str' antes de aplicar el reemplazo
    df['pases_completados_p'] = df['pases_completados_p'].astype(str).str.replace(',', '.').astype(float)

    # No se realiza esa conversión previa a 'str' y por ese motivo da 'Error'
    # df['pases_completados_p'] = df['pases_completados_p'].str.replace(',', '.').astype(float)

    # Modificar el proceso de agrupación
    df_agrupado = df.groupby('jugador', as_index=False).agg(
        {**{col: 'last' for col in columnas_sin_operar if col != 'posicion'},
        'posicion': obtener_posicion_mas_frecuente,
        **{col: 'sum' for col in columnas_a_sumar},
        **{col: 'mean' for col in columnas_a_mediar}}
    )

    df_agrupado['pases_completados_p'] = df_agrupado['pases_completados_p'].round(1)

    # Agrupar por 'jugador' y sumar las columnas correspondientes
    # df_agrupado = df.groupby('jugador', as_index=False)[columnas_a_sumar].sum()

    # Agrupar por 'jugador' y sumar las columnas correspondientes
    # df_agrupado = df.groupby('jugador', as_index=False)[columnas_a_mediar].mean().round(1)

    # Aplicar la función a la columna 'posición'
    df_agrupado['posicion'] = df_agrupado['posicion'].apply(mapear_posiciones)

    # Mostrar el DataFrame modificado
    # print(df_agrupado[['jugador', 'posicion']].head())

    return df_agrupado

#_________________________________________________________________

def generar_df_por_partido (df_agrupado):

    # Copiar el DataFrame agrupado
    df_por_partido = df_agrupado.copy()

    # Dividir la columna 'minutos' entre 90 para obtener el número de partidos jugados
    df_por_partido['partidos'] = df_por_partido['minutos'] / 90

    # Definir las columnas que deseas dividir por el número de partidos
    columnas_a_operar = [
        'goles', 'asistencias', 'penaltis_marcados', 'penaltis_intentados', 
        'total_tiros', 'tiros_a_puerta', 'amarillas', 'rojas', 'contactos', 'tackles', 
        'interceptaciones', 'bloqueos', 'goles_esp', 'goles_esp_np', 'goles_esp_asistidos', 
        'acciones_creacion_tiro', 'acciones_creacion_gol', 'pases_completados', 'pases_intentados',
        'pases_progresivos', 'conducciones', 'conducciones_progresivas', 'intentos_regate', 'regates_exitosos'
    ]

    # Dividir las columnas seleccionadas por el número de partidos jugados
    df_por_partido[columnas_a_operar] = df_por_partido[columnas_a_operar].div(df_por_partido['partidos'], axis=0).round(2)

    # Redondear a dos decimales
    df_por_partido['partidos'] = df_por_partido['partidos'].round(2)

    # Eliminar la columna 'minutos'
    df_por_partido = df_por_partido.drop(columns=['minutos'])

    # Reemplazar la columna 'minutos' con 'partidos' en la misma posición original
    # Insertar la columna 'partidos' en el lugar original de 'minutos'
    minutos_index = df_agrupado.columns.get_loc('minutos')
    df_por_partido.insert(minutos_index, 'partidos', df_por_partido.pop('partidos'))

    return df_por_partido

#_________________________________________________________________

def generar_df_consulta(df_agrupado):
    consulta = '''
    SELECT *
    FROM df_agrupado
    WHERE equipo = 'Barcelona'
    '''
    df_consulta = sqldf(consulta, locals())  # Pasar el entorno local para identificar el DataFrame
    return df_consulta

#_________________________________________________________________

def generar_df_agrupado_f(df_agrupado):

    max_minutos = df_agrupado['minutos'].max()

    # Filtro de minutos minimos
    min_minutos = max_minutos // 4  # division entera

    consulta = f'''
    SELECT *
    FROM df_agrupado
    WHERE minutos >= {min_minutos}
    '''

    df_agrupado_f = sqldf(consulta, locals())

    return df_agrupado_f

#_________________________________________________________________

def generar_df_metricas (df_agrupado_f):

    df_metricas = df_agrupado_f.copy()
    #df_metricas = pd.DataFrame(df_agrupado_f)

    df_metricas['jugador'] = df_agrupado_f['jugador']
    # df_metricas['TIRO'] = (df_agrupado_f['goles'] / df_agrupado_f['total_tiros'] * 100).round(1)
    df_metricas['TIRO'] = df_agrupado_f['tiros_a_puerta']
    # df_metricas['PASE'] = (df_agrupado_f['pases_completados'] / df_agrupado_f['pases_intentados'] * 100).round(1)
    df_metricas['PASE'] = df_agrupado_f['pases_completados_p']
    # df_metricas['REGATE'] = (df_agrupado_f['regates_exitosos'] / df_agrupado_f['intentos_regate'] * 100).round(1)
    df_metricas['REGATE'] = df_agrupado_f['regates_exitosos']
    df_metricas['CREACION'] = df_agrupado_f[['acciones_creacion_tiro', 'acciones_creacion_gol', 'pases_progresivos', 'conducciones_progresivas']].sum(axis=1)
    df_metricas['ATAQUE'] = df_agrupado_f[['goles', 'asistencias', 'tiros_a_puerta']].sum(axis=1)
    df_metricas['DEFENSA'] = (df_agrupado_f['pases_completados_p'] * (df_agrupado_f['interceptaciones'] + df_agrupado_f['bloqueos'] + df_agrupado_f['tackles']) / 100).round(1)

    # Visualizar el DataFrame con las nuevas métricas
    df_metricas = df_metricas[['jugador', 'equipo', 'posicion', 'TIRO', 'ATAQUE', 'REGATE', 'CREACION', 'PASE', 'DEFENSA']]

    return df_metricas

#_________________________________________________________________

def jugadores_similares(nombre_jugador, topn, df):
    """
    Dada la entrada de un jugador, calcula y devuelve los jugadores más similares.
    """
    # Seleccionar solo las columnas numéricas
    df_numeric = df.drop(columns=['jugador', 'equipo', 'numero', 'nacionalidad', 'posicion', 'edad'])

    # Normalizar los datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    
    # Crear DataFrame escalado con los jugadores como índice
    df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns, index=df['jugador'])
    
    # Calcular la matriz de similitud de coseno
    similarity_matrix = cosine_similarity(df_scaled)
    
    # Convertir la matriz a un DataFrame
    df_similarity = pd.DataFrame(similarity_matrix, index=df['jugador'], columns=df['jugador'])
    
    # Verificar si el jugador existe en los datos
    if nombre_jugador not in df_similarity.index:
        return f"El jugador {nombre_jugador} no fue encontrado."
    
    # Ordenar los jugadores por similitud
    similitudes = df_similarity[nombre_jugador].sort_values(ascending=False)
    
    # Excluir al jugador mismo y devolver los más similares
    jugadores_similares = similitudes.drop(nombre_jugador).head(topn)
    
    return jugadores_similares

#_________________________________________________________________

def generar_percentiles_jugador_df(jugador, df_agrupado_f):

    columnas_a_radiar = [
    'minutos', 'penaltis_marcados', 'penaltis_intentados', 
    'total_tiros', 'tiros_a_puerta', 'amarillas', 'rojas', 'contactos', 'tackles', 
    'interceptaciones', 'bloqueos', 'goles_esp', 'goles_esp_np', 'goles_esp_asistidos', 
    'acciones_creacion_tiro', 'acciones_creacion_gol', 'pases_completados', 'pases_intentados', 'pases_completados_p',
    'goles', 'asistencias', 'pases_progresivos', 'conducciones', 'conducciones_progresivas', 'intentos_regate', 'regates_exitosos'
    ]

    # Filtrar los datos del jugador
    jugador_data = df_agrupado_f[df_agrupado_f['jugador'] == jugador]

    # Obtener solo la primera posición del jugador
    # posicion_jugador = jugador_data['posicion'].values[0].split(',')[0]  # Tomar la primera posición

    # Obtener la posición del jugador ya calculada en 'df_agrupado_f'
    posicion_jugador = jugador_data['posicion'].values[0]  # Ya contiene la posición más frecuente

    # Filtrar los jugadores que juegan en esa primera posición
    # jugadores_misma_posicion = df_agrupado_f[df_agrupado_f['posicion'].str.contains(posicion_jugador)]

    # Filtrar los jugadores que juegan en esa posición
    jugadores_misma_posicion = df_agrupado_f[df_agrupado_f['posicion'] == posicion_jugador]

    # Calcular el percentil de cada variable para el jugador comparado con los otros de su posición
    percentiles_jugador = (jugadores_misma_posicion[columnas_a_radiar].rank(pct=True).loc[jugador_data.index] * 100).round(0)

    # Resetear el índice del DataFrame
    percentiles_jugador_df = percentiles_jugador.reset_index(drop=True)

    # Añadir el nombre del jugador como índice
    percentiles_jugador_df.index = [jugador]

    return percentiles_jugador_df, posicion_jugador

#_________________________________________________________________

def generar_percentiles_jugador_df_metricas(jugador, df_metricas):

    columnas_a_radiar = ['TIRO', 'ATAQUE', 'REGATE', 'CREACION', 'PASE', 'DEFENSA']

    # Filtrar los datos del jugador
    jugador_data = df_metricas[df_metricas['jugador'] == jugador]

    # Obtener solo la primera posición del jugador
    # posicion_jugador = jugador_data['posicion'].values[0].split(',')[0]  # Tomar la primera posición

    # Obtener la posición del jugador ya calculada en 'df_agrupado_f'
    posicion_jugador = jugador_data['posicion'].values[0]  # Ya contiene la posición más frecuente

    # Filtrar los jugadores que juegan en esa primera posición
    # jugadores_misma_posicion = df_agrupado_f[df_agrupado_f['posicion'].str.contains(posicion_jugador)]

    # Filtrar los jugadores que juegan en esa posición
    jugadores_misma_posicion = df_metricas[df_metricas['posicion'] == posicion_jugador]

    # Calcular el percentil de cada variable para el jugador comparado con los otros de su posición
    percentiles_jugador = (jugadores_misma_posicion[columnas_a_radiar].rank(pct=True).loc[jugador_data.index] * 100).round(0)

    # Resetear el índice del DataFrame
    percentiles_jugador_df = percentiles_jugador.reset_index(drop=True)

    # Añadir el nombre del jugador como índice
    percentiles_jugador_df.index = [jugador]

    valor_global = percentiles_jugador_df[['TIRO', 'PASE', 'REGATE', 'CREACION', 'ATAQUE', 'DEFENSA']].mean(axis=1).values[0].round(1)

    return percentiles_jugador_df, posicion_jugador, valor_global

#_________________________________________________________________

def obtener_top_jugadores_global(df_metricas, n=10):
    # Crear lista para almacenar resultados
    resultados = []
    
    # Calcular valor global para cada jugador
    for jugador in df_metricas['jugador']:
        _, _, valor_global = generar_percentiles_jugador_df_metricas(jugador, df_metricas)
        resultados.append({
            'jugador': jugador,
            'equipo': df_metricas[df_metricas['jugador'] == jugador]['equipo'].iloc[0],
            'posicion': df_metricas[df_metricas['jugador'] == jugador]['posicion'].iloc[0],
            'GLOBAL': valor_global
        })
    
    # Convertir a DataFrame y ordenar
    df_top = pd.DataFrame(resultados)
    df_top = df_top.sort_values('GLOBAL', ascending=False).head(n)
    return df_top