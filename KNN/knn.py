import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Cargar los datos desde el archivo CSV
data = pd.read_csv("iris.csv")

# Mezclar los datos
data = data.sample(frac=1)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X = data.iloc[:, :-1]  # Atributos
y = data.iloc[:, -1]  # Clases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Función Distancia euclidiana 
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Función k-NN
def k_nearest_neighbors(train_data, test_point, k):
    distances = []
    for index, row in train_data.iterrows():
        dist = euclidean_distance(test_point, row[:-1])
        distances.append((row, dist))
    distances.sort(key=lambda x: x[1])
    k_nearest = distances[:k]
    k_nearest_labels = [row.iloc[-1] for row, _ in k_nearest]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

best_k = None
best_accuracy = 0

#Buscar mejor valor de k
for k in range(1, 11):
    predictions = []

    for index, test_row in X_test.iterrows():
        predicted_class = k_nearest_neighbors(pd.concat([X_train, y_train], axis=1), test_row, k)
        predictions.append(predicted_class)

    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("Mejor valor de k:", best_k, '\n')

data = {"Clases Esperadas": y_test.values, 'Clases Predichas': predictions}
df = pd.DataFrame(data)
print(df)

accuracy = accuracy_score(y_test, predictions)
print(f"\nPrecisión del modelo con k = {best_k}: {accuracy * 100:.2f}%")