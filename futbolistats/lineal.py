# lineal.py

import numpy as np
import matplotlib.pyplot as plt

def visualizar_regresion_lineal(X, y, modelo_lineal, jugador, stat_a_predecir, jornadas_disputadas, jornadas_a_predecir):
    
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Datos reales')
    
    # Generar puntos para la línea de predicción
    x_pred = np.linspace(1, jornadas_disputadas + jornadas_a_predecir, jornadas_disputadas + jornadas_a_predecir)
    y_pred = modelo_lineal.predict(x_pred.reshape(-1, 1))
    
    plt.plot(x_pred, y_pred, color='red', label='Prediccion Regresion Lineal')
    plt.xlabel('Jornada')
    plt.ylabel(f'Acumulado de {stat_a_predecir.capitalize()}')
    plt.title(f'{jugador}: Predicción de {stat_a_predecir.capitalize()}')
    plt.legend()
    plt.grid()
    plt.xticks(range(1, (jornadas_disputadas+1) + jornadas_a_predecir))
    plt.tight_layout()
    
    return fig