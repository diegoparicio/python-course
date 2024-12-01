import torch
from torch import nn
import matplotlib.pyplot as plt

from pathlib import Path

def generar_datos(df_jugador_stat):

    # Convertir los datos a tensores de PyTorch
    x = torch.tensor(df_jugador_stat['jornada'].values, dtype=torch.float32)
    y = torch.tensor(df_jugador_stat['total_acumulado'].values, dtype=torch.float32)
    
    # Normalización Min-Max
    # x = (x - x.min()) / (x.max() - x.min())
    # y = (y - y.min()) / (y.max() - y.min())
    
    train_split = int(0.8 * len(x))
    x_train = x[:train_split]
    y_train = y[:train_split]

    x_test = x[train_split:]
    y_test = y[train_split:]

    #print("Datos de x_train: ", x_train)
    #print("\nDatos de y_train: ", y_train)
    #print("\nDatos de x_test: ", x_test)
    #print("\nDatos de y_test: ", y_test)

    return x_train, y_train, x_test, y_test

# Creando el modelo
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inicializar con valores más cercanos a 0
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float) * 0.1)
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

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


def entrenar_modelo(x_train, y_train, x_test, y_test, model, epochs=200, patience=3):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    best_test_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        y_pred = model(x_train)
        train_loss = loss_fn(y_pred, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Evaluación
        model.eval()
        with torch.inference_mode():
            test_pred = model(x_test)
            test_loss = loss_fn(test_pred, y_test)
            
            if epoch % 1 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(train_loss.item())
                test_loss_values.append(test_loss.item())
                # print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # print(f"\nDetención temprana en epoch {epoch}")
                print(f"Mejor pérdida de prueba: {best_test_loss:.4f} en el epoch {best_epoch}")
                model.load_state_dict(best_model_state)
                break

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
    model_loaded.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True))

    print("Cargado el modelo: ", model_loaded.state_dict())

    return model_loaded

#_________________________________________________________________

# Funcion para visualizar los datos
def visualizar_regresion_lineal_pytorch(x_train, y_train, x_test, y_test, jugador, stat, predictions=None, nuevos_datos=None, nuevas_predicciones=None, train=True):
    
    fig = plt.figure(figsize=(10, 5))
    
    plt.scatter(x_train, y_train, c="blue", marker='o', label="Datos de entrenamiento" if train else "Datos de testing")
    plt.scatter(x_test, y_test, c="orange", marker='o', label="Datos de testing")
    if predictions is not None:
        plt.scatter(x_test, predictions, c="green", marker='o', label="Predicciones")
    if nuevos_datos is not None and nuevas_predicciones is not None:
        plt.plot(nuevos_datos, nuevas_predicciones, c="red", label="Nuevas predicciones")
        plt.plot(nuevos_datos[0], nuevas_predicciones[0], color='red', marker='o')
        plt.plot(nuevos_datos[-1], nuevas_predicciones[-1], color='red', marker='o')
    
    plt.title(f'{jugador}: Predicción de {stat.capitalize()}')
    plt.xlabel('Jornada')
    plt.ylabel(f'Acumulado de {stat.capitalize()}')
    plt.legend()
    plt.grid()
    plt.xticks(range(1, int(nuevos_datos[-1].item()) + 1))
    plt.tight_layout()

    return fig