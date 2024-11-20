import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
# from logica import *

# Cargar el CSV de estadísticas de jugadores
# df = cargar_datos('laliga/database.csv')

#_________________________________________________________________

def calcular_jornadas_disputadas (df):

    # Encontrar el jugador con el máximo de minutos acumulados
    jugador_max_minutos = df.groupby('jugador')['minutos'].sum().idxmax()

    # Filtrar las filas del jugador con el máximo número de minutos
    df_jugador_max_minutos = df[df['jugador'] == jugador_max_minutos]

    # st.write(f"Jugador Max Minutos: {jugador_max_minutos}")

    # Contar el número de jornadas en las que aparece este jugador
    jornadas_disputadas = df_jugador_max_minutos.shape[0]

    # Mostrar el resultado en Streamlit
    # st.write(f"Jornadas Disputadas: {jornadas_disputadas}")

    return jornadas_disputadas

#_________________________________________________________________

def generar_df_jugador_stat(df, jornadas_disputadas):

    # Convertir la columna de fecha a datetime si es necesario
    df['fecha'] = pd.to_datetime(df['fecha'])

    # jugador_base = "Raphinha"

    # Menú para seleccionar un jugador
    # jugador = st.selectbox("Selecciona un jugador:", df['jugador'].unique(), index=df['jugador'].tolist().index(jugador_base)) # .unique()

    # Crear selección de jugador en Streamlit
    jugador = st.selectbox("Selecciona un jugador", df['jugador'].unique())

    # Filtrar datos del jugador seleccionado
    df_jugador = df[df['jugador'] == jugador]

    # Selección de predicción: Goles o Asistencias
    stat_a_predecir = (st.selectbox("Selecciona la estadística para predecir", ["Goles", "Asistencias"])).lower()

    # Preparar los datos en el formato esperado
    df_jugador_stat = df_jugador[['fecha', stat_a_predecir]].rename(columns={'fecha': 'ds', stat_a_predecir: 'y'})

    # Agregar la columna 'jornada' que incrementa de 1 en cada fila
    df_jugador_stat['jornada'] = range(1, len(df_jugador_stat) + 1)

    # Filtrar los datos hasta la jornada 10
    # df_jugador_stat = df_jugador_stat[df_jugador_stat['jornada'] <= 10]
    df_jugador_stat = df_jugador_stat[df_jugador_stat['jornada'] <= jornadas_disputadas]

    # Calcular el total acumulado
    df_jugador_stat['total_acumulado'] = df_jugador_stat['y'].cumsum()

    # Mostrar los datos actuales del jugador seleccionado  <--- DATASET 1
    # st.write(f"Datos de {stat_a_predecir} de {jugador}")
    # st.write(df_jugador_stat)

    return df_jugador_stat, jugador, stat_a_predecir

#_________________________________________________________________

def generar_pred_df(df_jugador_stat, jornadas_disputadas):

    # Configurar el número de jornadas para predecir usando un slider
    # Usar el valor guardado en session_state como value inicial
    jornadas_a_predecir = st.slider(
        "Selecciona el número de jornadas a predecir", 
        min_value=1, 
        max_value=(38 - jornadas_disputadas), 
        value=(38 - jornadas_disputadas)
    )

    # Crear un DataFrame para las jornadas futuras
    # jornadas_futuras = pd.DataFrame({'jornada': range(11, 11 + jornadas_a_predecir)})
    jornadas_futuras = pd.DataFrame({'jornada': range(jornadas_disputadas+1, (jornadas_disputadas+1) + jornadas_a_predecir)})

    # Calcular la tasa promedio por jornada
    tasa_promedio = df_jugador_stat['y'].mean()

    # Calcular las predicciones basadas en la tasa promedio
    predicciones = [tasa_promedio] * jornadas_a_predecir  # Usar la misma predicción para cada jornada futura

    # Crear un DataFrame con las predicciones
    pred_df = pd.DataFrame({
        'jornada': jornadas_futuras['jornada'],
        'prediccion': predicciones  # Mantener predicciones constantes
    })

    # Limitar la columna 'prediccion' a 2 decimales
    pred_df['prediccion'] = pred_df['prediccion'].round(3)

    # Calcular los valores acumulados de las predicciones
    total_acumulado_previo = df_jugador_stat['total_acumulado'].iloc[-1]
    pred_df['total_acumulado'] = (pred_df['prediccion'].cumsum() + total_acumulado_previo)  # .round(0) -> si lo quitas, linea recta

    # Mostrar el DataFrame con las predicciones  <--- DATASET 2
    # st.dataframe(pred_df)

    return pred_df, jornadas_a_predecir

#_________________________________________________________________

def visualizar_prediccion_promedio(df_jugador_stat, pred_df, stat, jornadas_disputadas, jornadas_a_predecir, jugador):

    # Visualización de los totales acumulados
    fig = plt.figure(figsize=(10, 5))
    
    plt.plot(df_jugador_stat['jornada'], df_jugador_stat['total_acumulado'], label='Datos reales', color='blue')

    plt.plot(pred_df['jornada'], pred_df['total_acumulado'], label='Predicción Promedio', color='red')
    # Agregar marker solo en el primer y último punto
    plt.plot(pred_df['jornada'].iloc[0], pred_df['total_acumulado'].iloc[0], color='red', marker='o')
    plt.plot(pred_df['jornada'].iloc[-1], pred_df['total_acumulado'].iloc[-1], color='red', marker='o')

    plt.title(f'{jugador}: Predicción de {stat.capitalize()}')
    plt.xlabel('Jornada')
    plt.ylabel(f'Acumulado de {stat.capitalize()}')
    plt.legend()
    plt.grid()
    # plt.xticks(range(1, 11 + jornadas_a_predecir))  # Cambia según el rango de jornadas que estés utilizando
    plt.xticks(range(1, (jornadas_disputadas+1) + jornadas_a_predecir))  # Cambia según el rango de jornadas que estés utilizando
    plt.tight_layout()

    # Mostrar la gráfica en Streamlit
    return fig

#_________________________________________________________________