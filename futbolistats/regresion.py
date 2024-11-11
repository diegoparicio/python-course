import torch
from torch import nn
import matplotlib.pyplot as plt

from pathlib import Path

def generar_datos(df_jugador_stat):

    # Convertir los datos a tensores de PyTorch
    x = torch.tensor(df_jugador_stat['jornada'].values, dtype=torch.float32)
    y = torch.tensor(df_jugador_stat['total_acumulado'].values, dtype=torch.float32)

    train_split = int(0.8 * len(x))
    x_train = x[:train_split]
    y_train = y[:train_split]

    x_test = x[train_split:]
    y_test = y[train_split:]

    print("Datos de x_train: ", x_train)
    print("\nDatos de y_train: ", y_train)
    print("\nDatos de x_test: ", x_test)
    print("\nDatos de y_test: ", y_test)

    return x_train, y_train, x_test, y_test


# Funcion para visualizar los datos
def plot_predictions(x_train, y_train, x_test, y_test, predictions = None, train=True):
    
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(x_train, y_train, c="blue", label="Training data" if train else "Testing data")
    plt.scatter(x_test, y_test, c="red", label="Testing data")
    if predictions is not None:
        plt.scatter(x_test, predictions, c="green", label="Predictions")
    plt.legend()
    # plt.show()

    return fig

# plot_predictions(x_train, y_train, x_test, y_test)


# Creando el modelo
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias # Formula de la regresion lineal

def crear_modelo():
    torch.manual_seed(42)
    model = LinearRegressionModel()
    params = list(model.parameters())

    return model

def hacer_predicciones(x_test, model):
    # Hacer predicciones
    with torch.inference_mode():
        y_preds = model(x_test)
        y_preds = torch.relu(y_preds)  # Aplicar ReLU para asegurar que las predicciones sean >= 0

    return y_preds


def entrenar_modelo(x_train, y_train, x_test, y_test, model):
    # Funcion de perdida
    loss_fn = nn.L1Loss()

    # Optimizador 
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    torch.manual_seed(42)

    epochs = 100  # ciclos de entrenamiento

    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    # Entrenamiento
    for epoch in range(epochs):
        model.train()

        # 1. Hacer predicciones
        y_pred = model(x_train)

        # 2. Calcular la perdida
        loss = loss_fn(y_pred, y_train)

        # 3. Optimizar las predicciones
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        model.eval()
        with torch.inference_mode():
            test_pred = model(x_test)

            test_loss = loss_fn(test_pred, y_test.type(torch.float))

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")

    return epoch_count, train_loss_values, test_loss_values

def graficar_perdidas(epoch_count, train_loss_values, test_loss_values):
    fig = plt.figure(figsize=(10, 7))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(epoch_count, train_loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.legend(fontsize=18)
    
    return fig


# Ver los parametros que aprendio el modelo
# print("Parametros aprendidos por el modelo (weight = pendiente y bias = intercepto):")
# print(model.state_dict())

#print("\n Valores originales:")
#print(f"Pendiente (m): {val1} | Intercepto (b): {val2}")

#_________________________________________________________________

# model.eval()


#_________________________________________________________________

# Guardar el modelo

def guardar_modelo(model):
    # Crear directorio para guardar el modelo
    MODEL_PATH = Path("futbolistats")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Directorio del modelo
    MODEL_NAME = "modelo.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Guardar el modelo
    print(f"Guardando el modelo en {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

#_________________________________________________________________

# Cargar el modelo

def cargar_modelo():
    # Definir el directorio del modelo dentro de la función
    MODEL_PATH = Path("futbolistats")
    MODEL_NAME = "modelo.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    model_loaded = LinearRegressionModel()
    model_loaded.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    print("Cargado el modelo: ", model_loaded.state_dict())

    return model_loaded

#_________________________________________________________________

# Funcion para visualizar los datos
def plot_predictions_new(x_train, y_train, x_test, y_test, jugador, stat, predictions=None, nuevos_datos=None, nuevas_predicciones=None, train=True):
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x_train, y_train, c="blue", marker='o', label="Datos de entrenamiento" if train else "Datos de testing")
    plt.plot(x_test, y_test, c="red", marker='o', label="Datos de testing")
    if predictions is not None:
        plt.plot(x_test, predictions, c="green", marker='o', label="Predicciones")
    if nuevos_datos is not None and nuevas_predicciones is not None:
        plt.plot(nuevos_datos, nuevas_predicciones, c="orange", linestyle='--', label="Nuevas predicciones")
        plt.plot(nuevos_datos[0], nuevas_predicciones[0], color='orange', marker='o')
        plt.plot(nuevos_datos[-1], nuevas_predicciones[-1], color='orange', marker='o')
    
    plt.title(f'{jugador}: Predicción de {stat.capitalize()}')
    plt.xlabel('Jornada')
    plt.ylabel(f'Acumulado de {stat.capitalize()}')
    plt.legend()
    plt.grid()
    # plt.show()

    return fig