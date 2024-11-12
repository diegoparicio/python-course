# app.py

import streamlit as st
import pandas as pd
import numpy as np
from ayudantes import *
from logica import *
from poison import *

from regresion import *

#_________________________________________________________________

# Título de la aplicación
st.title("FutboliStats: Liga, Premier, Serie A")
st.write("Creado por @diegoparicio")

#_________________________________________________________________
# Agrupar los datos de jugadores

st.header("Estadísticas Globales")

df_agrupado = generar_df_agrupado(df)

# Mostrar el DataFrame en Streamlit
st.dataframe(df_agrupado)

#_________________________________________________________________

st.header("Estadísticas por Partido (90 Minutos)")

df_por_partido = generar_df_por_partido(df_agrupado)

st.dataframe(df_por_partido)

#_________________________________________________________________

# st.header("Estadísticas Barcelona")

df_consulta = generar_df_consulta(df_agrupado)

# st.dataframe(df_consulta)

#_________________________________________________________________

# DataFrame Filtrado
df_agrupado_f = generar_df_agrupado_f(df_agrupado)

#_________________________________________________________________

st.header("Estadísticas Métricas Depuradas (F)")
st.write("F = filtrado por minutos mínimos jugados")

df_metricas = generar_df_metricas(df_agrupado_f)

st.dataframe(df_metricas)

#_________________________________________________________________

st.header("Jugadores Similares (F)")

#jugador_base = "Raphinha"
# jugador = st.selectbox("Selecciona un jugador:", df_agrupado_f['jugador'], index=df_agrupado_f['jugador'].tolist().index(jugador_base))

jugador = st.selectbox("Selecciona un jugador:", df_agrupado_f['jugador'])

# Slider para el número de jugadores similares a mostrar
topn = st.slider("Número de jugadores similares a mostrar:", 1, 10, 5)

tipo_similitud = (st.selectbox("Filtro por posición:", ["Cualquiera", "Identica"])).lower()

if tipo_similitud == "identica":

    # Obtener la posición del jugador seleccionado
    posicion_jugador = df_agrupado_f[df_agrupado_f['jugador'] == jugador]['posicion'].iloc[0]

    # Filtrar el dataframe para jugadores de la misma posición
    df_misma_posicion = df_agrupado_f[df_agrupado_f['posicion'] == posicion_jugador]

    # Calcular los jugadores más similares (solo entre jugadores de la misma posición)
    resultados_similares = jugadores_similares(jugador, topn, df_misma_posicion)

    # Mostrar resultados
    st.write(f"Jugadores similares a {jugador}, en posición: {posicion_jugador}:")

elif tipo_similitud == "cualquiera":

    # Calcular los jugadores más similares
    resultados_similares = jugadores_similares(jugador, topn, df_agrupado_f)

    # Mostrar resultados
    st.write(f"Jugadores similares a {jugador}:")

st.write(resultados_similares)

# Mostrar gráfico de similitud
fig = generar_grafico_similitud(resultados_similares, jugador)
st.pyplot(fig)

#_________________________________________________________________

st.header("Gráficos Radiales (F)")

# J1_base = "Raphinha"
# J2_base = "Lamine Yamal"

opcion_grafico = st.selectbox("Seleccione el tipo de gráfico radial. (Simple: métricas depuradas)", ["Simple", "Detallado"])

# J1 = st.selectbox("Selecciona el primer jugador:", df_agrupado_f['jugador'], index=df_agrupado_f['jugador'].tolist().index(J1_base), key="jugador1")
# J2 = st.selectbox("Selecciona el segundo jugador:", df_agrupado_f['jugador'], index=df_agrupado_f['jugador'].tolist().index(J2_base), key="jugador2")

J1 = st.selectbox("Selecciona un jugador 1:", df_agrupado_f['jugador'])
J2 = st.selectbox("Selecciona un jugador 2:", df_agrupado_f['jugador'])

if opcion_grafico == "Simple":

    df_percen_J1, pos_J1, valor_global_J1 = generar_percentiles_jugador_df_metricas(J1, df_metricas)
    df_percen_J2, pos_J2, valor_global_J2 = generar_percentiles_jugador_df_metricas(J2, df_metricas)

    df_final = pd.concat([df_percen_J1, df_percen_J2])
    valores_globales = [valor_global_J1, valor_global_J2]
    df_final['GLOBAL'] = valores_globales

    st.dataframe(df_final)

    # NO SE HACE ASÍ
    #globales = pd.concat([valor_global_J1, valor_global_J2])
    #st.dataframe(globales)

    
else:
    df_percen_J1, pos_J1 = generar_percentiles_jugador_df(J1, df_agrupado_f)
    df_percen_J2, pos_J2 = generar_percentiles_jugador_df(J2, df_agrupado_f)

    df_final = pd.concat([df_percen_J1, df_percen_J2])
    st.dataframe(df_final)


# Crear el gráfico de radar
fig = generar_grafico_radar_1(df_percen_J1, J1, pos_J1)

# Mostrar el gráfico en Streamlit
# st.pyplot(fig)
# st.text("Basado en los percentiles respecto a los jugadores que ocupan su misma posición")

# Crear el gráfico de radar
fig = generar_grafico_radar_2(df_percen_J1, df_percen_J2, J1, J2, pos_J1, pos_J2)
st.pyplot(fig)

st.write("Basado en los percentiles respecto a los jugadores que ocupan su misma posición")

#_________________________________________________________________

st.header("Predicción Goles y Asistencias")

#_________________________________________________________________

# Calcular el máximo de jornadas disputadas entre todas las ligas
jornadas_disputadas_max = calcular_jornadas_disputadas(df)

# Valor base para el slider
jornadas_base = 10

# Slider para las jornadas disputadas
jornadas_disputadas = st.slider("Jornadas Disputadas", min_value=5, max_value=jornadas_disputadas_max, value=jornadas_base)

# Selectbox para las jornadas disputadas
# jornadas_disputadas = st.selectbox("Jornadas Disputadas:", range(int(5), jornadas_disputadas_max+1, 1))

df_jugador_stat, jugador, stat_a_predecir = generar_df_jugador_stat(df, jornadas_disputadas)

pred_df, jornadas_a_predecir = generar_pred_df(df_jugador_stat, jornadas_disputadas)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

st.subheader("Predicción con Poison")

fig = generar_grafico_predicion(df_jugador_stat, pred_df, stat_a_predecir, jornadas_disputadas, jornadas_a_predecir, jugador)
st.pyplot(fig)

#_________________________________________________________________

st.subheader("Predicción con Regresión Lineal (PyTorch)")

x_train, y_train, x_test, y_test = generar_datos(df_jugador_stat)

# fig = plot_predictions(x_train, y_train, x_test, y_test)
# st.pyplot(fig)

model = crear_modelo()

# Hacer predicciones antes de entrenar el modelo
# y_preds = hacer_predicciones(x_test, model)
# print("Predicciones de PyTorch: ", y_preds)

epoch_count, train_loss_values, test_loss_values = entrenar_modelo(x_train, y_train, x_test, y_test, model)

# fig = graficar_perdidas(epoch_count, train_loss_values, test_loss_values)
# st.pyplot(fig)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

# Hacer predicciones antes de guardar el modelo
y_preds = hacer_predicciones(x_test, model)

# fig = plot_predictions(x_train, y_train, x_test, y_test, y_preds)
# st.pyplot(fig)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

# Guardar el modelo
guardar_modelo(model)
    
# Cargar el modelo
model_loaded = cargar_modelo()

# Verificar si el modelo se cargó correctamente
if model_loaded is None:
    raise ValueError("El modelo no se cargó correctamente. Verifica la función cargar_modelo.")

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

# Hacer predicciones con el modelo cargado
y_preds_loaded = hacer_predicciones(x_test, model_loaded)

# fig = plot_predictions(x_train, y_train, x_test, y_test, y_preds_loaded)
# st.pyplot(fig)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

print(y_preds == y_preds_loaded)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

# Nuevos datos para predecir
nuevos_datos = torch.tensor([float(x) for x in range(int(jornadas_disputadas+1), (jornadas_disputadas+1) + jornadas_a_predecir, 1)])  # Sí, incluye hasta 38.0 porque range(n) va de 0 a n-1

# Hacer predicciones con el modelo cargado
nuevas_predicciones = hacer_predicciones(nuevos_datos, model_loaded)

print("Nuevas predicciones: ", nuevas_predicciones)

fig = plot_predictions_new(x_train, y_train, x_test, y_test, jugador, stat_a_predecir,predictions=y_preds_loaded, nuevos_datos=nuevos_datos, nuevas_predicciones=nuevas_predicciones)
st.pyplot(fig)

#_________________________________________________________________