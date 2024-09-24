from PIL import Image
import os
import numpy as np

# Ruta a la carpeta que contiene las imágenes
folder_path = 'animales'

# Lista para almacenar las imágenes cargadas
image_list = []

# Lista para almacenar las categorias por numeros para que la maquina pueda reconocerlos
image_diferenciada = []

# Recorre todos los archivos de la carpeta
for filename in os.listdir(folder_path):

    if 'perro' in filename:
        image_diferenciada.append(0)
    elif 'persona' in filename:
        image_diferenciada.append(1)
    elif 'hipopotamo' in filename:
        image_diferenciada.append(2)

    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(folder_path, filename) # fucion ruta de la carpeta de las imagenes + el recorrido activo del for filename activamente
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Redimensiona la imagen
        img_array = np.array(img) / 255.0  # Normaliza los valores de píxeles
        image_list.append(img_array)
        print(f'{filename} | nomralizacion y redimension {np.array(img).shape}')
print('')
print(f'Lista categorizada => {image_diferenciada}')
print(f"Tamaño de image_list: {len(image_list)}")
print(f"Tamaño de image_diferenciada: {len(image_diferenciada)}")

from sklearn.model_selection import train_test_split
# test_size=0.2: Indica que el 20% de los datos se utilizarán como conjunto de prueba, y el 80% restante como conjunto de entrenamiento.

# generacion de test_size=0.2: Indica que el 20% de los datos se utilizarán como conjunto de prueba, y el 80% restante como conjunto de entrenamiento.

# random_state=42: Fija la semilla del generador aleatorio para que la división sea reproducible. El número 42 es arbitrario.

X_train, X_test, y_train, y_test = train_test_split(image_list, image_diferenciada, test_size=0.3, random_state=42, stratify=image_diferenciada)

import keras
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Flatten # type: ignore

#Se establece una semilla para evitar distintos resultados
keras.utils.set_random_seed(812)
#Se establece un modelo neuronal
model = Sequential()
model.add(Flatten(input_shape=(224, 224, 3)))  # Aplanar la imagen
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax')) #3 es el numero de clases

#Se compila el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Verificamos la estructura de salida esperada para entrenamiento
y_train

#Verificamos shape de imagenes
import numpy as np

for i, image in enumerate(X_train):
    print(f"Shape of image {i}: {np.array(image).shape}")

#Cambiamos la forma
X_train = np.array(X_train)
y_train = np.array(y_train)

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_train

X_train.shape

#Verificamos el cambio
y_train

y_train.shape

# Ajustamos("Entrenamos") el modelo al conjunto de datos cada que se ejecuta este; el modelo nos devulve el accuracy mas preciso; La pérdida (loss) es una medida ve que tan mal está el modelo en esa época. Un valor de 1.1341 indica cuán lejos está el modelo de las predicciones correctas, basándose en la función de pérdida que estás utilizando (en este caso, categorical_crossentropy).

%time model.fit(X_train,y_train,epochs=20)

%time y_train

import numpy as np

# Investigate the shapes of elements within X_test
for i, item in enumerate(X_test):
    print(f"Shape of item {i}: {np.array(item).shape}")

# Option 1: Pad images with fewer channels (if applicable)
target_channels = 3  # Set the desired number of channels
X_test_padded = []
for item in X_test:
    item_array = np.array(item)
    if item_array.shape[-1] < target_channels:
        padding_needed = target_channels - item_array.shape[-1]
        padding = [(0, 0)] * (item_array.ndim - 1) + [(0, padding_needed)]
        item_padded = np.pad(item_array, padding, mode='constant')
        X_test_padded.append(item_padded)
    else:
        X_test_padded.append(item_array)

X_test = np.array(X_test_padded)
y_test = np.array(y_test)


if len(y_test.shape) == 0:
    y_test = y_test.reshape(1,)

print(y_test.shape)
print(X_test.shape)

# Verificamos su forma
y_test

# Usamos el modelo para predecir con el conjunto de prueba
y_pred = model.predict(X_test) # Esto genera una salida predicha

#Revisamos el resultado
y_pred

#Redondeamos los resultados
y_pred=np.round(y_pred)
y_pred

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score


y_pred = np.argmax(y_pred, axis=1)
# Accuracy
# Métricas de clasificación
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Calculamos precision, recall y f1 para cada clase
precision = precision_score(y_test, y_pred, average=None)
precision_avg = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average=None)
recall_avg = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average=None)
f1_avg = f1_score(y_test, y_pred, average="macro")
print(f"Exactitud: {accuracy}")
print(f"Precisión por clase: {precision} -> {precision_avg}")
print(f"Sensibilidad por clase: {recall} -> {recall_avg}")
print(f"F1 Score por clase: {f1} -> {f1_avg}")

from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.display_labels = ['Perro','Persona','Hipopotamo']
disp.plot()

y_test

y_pred

X_test[2]

import matplotlib.pyplot as plt
img_array = (X_test[2] * 255).astype(np.uint8)  # escala de 0-255 y conversion a 8-bit integer
img = Image.fromarray(img_array)
plt.imshow(img)

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score 
from tkinter import filedialog
from PIL import Image

# Abrir el cuadro de diálogo de archivos para seleccionar la imagen
ruta_imagen = 'C:/Users/mrpic/Downloads/perro-Chihuahua-1.jpg'

print(f"Has seleccionado la imagen: {ruta_imagen}")

# Cargar la imagen seleccionada y preprocesarla
img = Image.open(ruta_imagen)  # Cambia el tamaño según lo que usaste para entrenar el modelo
img = img.resize((224, 224)) # Redimensiona la imagen
img_array = np.array(img) / 255.0  # Convertir la imagen a un array numpy
img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra para que tenga forma (1, 224, 224, 3)

# Hacer la predicción usando el modelo entrenado
prediccion = model.predict(img_array)

# Decodificar la predicción
clases = ['perro', 'persona', 'hipopotamo']  # Ajusta según las clases de tu modelo
prediccion_clase = np.argmax(prediccion, axis=1)  # Obtener la clase con mayor probabilidad

# Mostrar la predicción
print(f"Predicción: {clases[prediccion_clase[0]]}")

# Opcional: Métricas de evaluación si tienes etiquetas reales y predicciones (si estás evaluando un conjunto de prueba)
# Nota: Esto se usa para evaluación después del entrenamiento, no durante la predicción individual de una imagen.
y_pred = model.predict(X_test)  # Usado para un conjunto de prueba con etiquetas
y_test = [...]  # Si tienes las etiquetas reales del conjunto de prueba

# Métricas (descomentar si quieres usarlas)
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
# print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
# print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")