import torch
from torch import nn
import matplotlib.pyplot as plt

#_________________________________________________________________

# Generando datos
val1 = 0.7  # Pendiente (m)
val2 = 1    # Intercepto (b)

start = 0   # Inicio del rango
end = 1    # Fin del rango
step = 0.02 # Paso

x = torch.arange(start, end, step)     # Generando los inputs (x)
y = val1 * x + val2                    # Generando los targets (y)   

# Separando los datos en conjuntos de entrenamiento y prueba
train_split = int(0.8 * len(x))
x_train = x[:train_split]
y_train = y[:train_split]

x_test = x[train_split:]
y_test = y[train_split:]

'''
print("Forma de datos x_train: ", x_train.shape)
print("\nForma de datos y_train: ", y_train.shape)
print("\nForma de datos x_test: ", x_test.shape)
print("\nForma de datos y_test: ", y_test.shape)
'''

# Funcion para visualizar los datos
def plot_predictions(x_train, y_train, x_test, y_test, predictions = None, train=True):
    plt.scatter(x_train, y_train, c="blue", label="Training data" if train else "Testing data")
    plt.scatter(x_test, y_test, c="red", label="Testing data")
    if predictions is not None:
        plt.scatter(x_test, predictions, c="green", label="Predictions")
    plt.legend()
    plt.show()

# plot_predictions(x_train, y_train, x_test, y_test)

#_________________________________________________________________

# Creando el modelo
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias # Formula de la regresion lineal

torch.manual_seed(42)
model = LinearRegressionModel()
params = list(model.parameters())

# print(params)

#_________________________________________________________________

# Hacer predicciones
with torch.inference_mode():
    y_preds = model(x_test)

print("Predicciones de PyTorch: ", y_preds)

plot_predictions(x_train, y_train, x_test, y_test, predictions=y_preds)

#_________________________________________________________________

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

#_________________________________________________________________

# Graficar perdidas

plt.figure(figsize=(10, 7))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.legend(fontsize=18)
plt.show()

#_________________________________________________________________

# Ver los parametros que aprendio el modelo
print("Parametros aprendidos por el modelo (weight = pendiente y bias = intercepto):")
print(model.state_dict())
print("\n Valores originales:")
print(f"Pendiente (m): {val1} | Intercepto (b): {val2}")

#_________________________________________________________________

model.eval()
with torch.inference_mode():
    y_preds = model(x_test)

plot_predictions(x_train, y_train, x_test, y_test, predictions=y_preds)

#_________________________________________________________________

# Guardar el modelo

from pathlib import Path

# Crear directorio para guardar el modelo
MODEL_PATH = Path("pytorch")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Directorio del modelo
MODEL_NAME = "modelo.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Guardar el modelo
print(f"Guardando el modelo en {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

#_________________________________________________________________

# Cargar el modelo

model_loaded = LinearRegressionModel()
model_loaded.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(model_loaded.state_dict())

#_________________________________________________________________

# Hacer predicciones con el modelo cargado

model_loaded.eval()
with torch.inference_mode():
    model_loaded_y_preds = model_loaded(x_test)

# plot_predictions(x_train, y_train, x_test, y_test, predictions=model_loaded_y_preds)

print(y_preds == model_loaded_y_preds)

#_________________________________________________________________

# Nuevos datos para predecir
nuevos_datos = torch.tensor([1.5, 2.0, 2.5])  # Ejemplo de nuevos valores de entrada

# Hacer predicciones con el modelo cargado
with torch.inference_mode():
    nuevas_predicciones = model_loaded(nuevos_datos)

print("Nuevas predicciones: ", nuevas_predicciones)


# Funcion para visualizar los datos
def plot_predictions_new(x_train, y_train, x_test, y_test, predictions=None, nuevos_datos=None, nuevas_predicciones=None, train=True):
    plt.scatter(x_train, y_train, c="blue", label="Training data" if train else "Testing data")
    plt.scatter(x_test, y_test, c="red", label="Testing data")
    if predictions is not None:
        plt.scatter(x_test, predictions, c="green", label="Predictions")
    if nuevos_datos is not None and nuevas_predicciones is not None:
        plt.scatter(nuevos_datos, nuevas_predicciones, c="purple", label="New Predictions")
    plt.legend()
    plt.show()

plot_predictions_new(x_train, y_train, x_test, y_test, predictions=y_preds, nuevos_datos=nuevos_datos, nuevas_predicciones=nuevas_predicciones)