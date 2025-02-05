import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Modelos
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# Métricas
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

import kagglehub
import pathlib

# Cargar datos
@st.cache_data
def cargar_datos():
    eltiempo_csv = pathlib.Path(kagglehub.dataset_download('zeeshier/weather-forecast-dataset')) / "weather_forecast_data.csv"
    datos = pd.read_csv(eltiempo_csv)
    return datos

datos = cargar_datos()

# Mostrar información de los datos
# st.write("## Información de los Datos")
# st.write(datos.info())
# st.write("## Valores Nulos")
# st.write(datos.isnull().sum())

# Eliminar valores nulos
datos.dropna(inplace=True)

# Codificar variables categóricas
label_encoders = {}
columnas_a_codificar = ['Rain']
for columna in columnas_a_codificar:
    le = LabelEncoder()
    datos[columna + '_codificada'] = le.fit_transform(datos[columna])
    label_encoders[columna] = le

st.image('rainforecast/media/portada.png', use_container_width=True)    
st.write("@diegoparicio")

st.subheader("## Datos de clima registrados (head)")
st.write(datos.head())

# Mapa de calor de correlación
st.subheader("## Mapa de Calor de Correlación")
plt.figure(figsize=(12,6))
sns.heatmap(datos.drop('Rain',axis=1).corr(),annot=True)
st.pyplot(plt)

# Gráficos de dispersión
st.subheader("## Gráficos de Dispersión")
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title('Temperatura y Humedad')
sns.scatterplot(x=datos['Temperature'],y=datos['Humidity'],hue=datos['Rain_codificada'])
plt.subplot(1,2,2)
plt.title('Temperatura y Humedad')
sns.kdeplot(x=datos['Temperature'],y=datos['Humidity'],hue=datos['Rain_codificada'])
st.pyplot(plt)

plt.figure(figsize=(10,4))
plt.title('Velocidad del Viento y Nubosidad')
sns.scatterplot(x=datos['Wind_Speed'],y=datos['Cloud_Cover'],hue=datos['Rain_codificada'])
st.pyplot(plt)

columnas = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
for i in columnas:
    plt.figure(figsize=(6,3))
    sns.scatterplot(x=datos[i],y=datos['Rain_codificada'])
    st.pyplot(plt)

# Dividir datos
X = datos.drop(['Rain','Rain_codificada'],axis=1)
y = datos['Rain_codificada']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Regresión Logística
st.subheader("## Regresión Logística")
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)
y_prediccion = modelo.predict(X_test)
precision = accuracy_score(y_test, y_prediccion)
st.write(f'La precisión de la Regresión Logística es {precision:.4f}')
matriz_confusion = confusion_matrix(y_test, y_prediccion)
plt.title('Matriz de Confusión')
sns.heatmap(matriz_confusion, annot=True, cmap='Blues')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Exacta')
st.pyplot(plt)

# KNeighborsClassifier
st.subheader("## KNeighborsClassifier")
modelo = KNeighborsClassifier()
modelo.fit(X_train, y_train)
y_prediccion = modelo.predict(X_test)
precision = accuracy_score(y_test, y_prediccion)
st.write(f'La precisión de KNeighborsClassifier es {precision:.4f}')
matriz_confusion = confusion_matrix(y_test, y_prediccion)
plt.title('Matriz de Confusión')
sns.heatmap(matriz_confusion, annot=True, cmap='Blues')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Exacta')
st.pyplot(plt)

# Curvas de Validación para KNeighborsClassifier
st.subheader("## Curvas de Validación para KNeighborsClassifier")
grado = np.arange(1, 21)
train_score, val_score = validation_curve(KNeighborsClassifier(n_neighbors=5), X, y, param_name='n_neighbors', param_range=grado, cv=5)
plt.title('Curvas de Validación de KNeighborsClassifier')
plt.plot(grado, train_score.mean(axis=1), label='training score')
plt.plot(grado, val_score.mean(axis=1), label='validation score')
plt.legend()
st.pyplot(plt)

# DecisionTreeClassifier
st.subheader("## DecisionTreeClassifier")
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
y_prediccion = modelo.predict(X_test)
precision = accuracy_score(y_test, y_prediccion)
st.write(f'La precisión del Árbol de Decisión es {precision:.4f}')
matriz_confusion = confusion_matrix(y_test, y_prediccion)
plt.title('Matriz de Confusión')
sns.heatmap(matriz_confusion, annot=True, cmap='Blues')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Exacta')
st.pyplot(plt)

# RandomForestClassifier
st.subheader("## RandomForestClassifier")
modelo = RandomForestClassifier(n_estimators=10)
modelo.fit(X_train, y_train)
y_prediccion = modelo.predict(X_test)
precision = accuracy_score(y_test, y_prediccion)
st.write(f'La precisión de RandomForestClassifier es {precision:.4f}')
matriz_confusion = confusion_matrix(y_test, y_prediccion)
plt.title('Matriz de Confusión')
sns.heatmap(matriz_confusion, annot=True, cmap='Blues')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Exacta')
st.pyplot(plt)

# Curvas de Validación para RandomForestClassifier
st.subheader("## Curvas de Validación para RandomForestClassifier")
grado = np.arange(1, 21)
train_score, val_score = validation_curve(RandomForestClassifier(), X, y, param_name='n_estimators', param_range=grado, cv=5)
plt.title('Curvas de Validación de RandomForestClassifier')
plt.plot(grado, train_score.mean(axis=1), label='training score')
plt.plot(grado, val_score.mean(axis=1), label='validation score')
plt.legend()
st.pyplot(plt)

# GaussianNB
st.subheader("## GaussianNB")
modelo = GaussianNB()
modelo.fit(X_train, y_train)
y_prediccion = modelo.predict(X_test)
precision = accuracy_score(y_test, y_prediccion)
st.write(f'La precisión de GaussianNB es {precision:.4f}')
matriz_confusion = confusion_matrix(y_test, y_prediccion)
plt.title('Matriz de Confusión')
sns.heatmap(matriz_confusion, annot=True, cmap='Blues')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Exacta')
st.pyplot(plt)

# Regresión Lineal
st.subheader("## Regresión Lineal")
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_prediccion = modelo.predict(X_test)
st.write(f'Puntuación R2: {r2_score(y_test, y_prediccion):.4f}')