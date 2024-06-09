#Proyecto 1. Convertir Celsius a fahrenheit
#El objetivo es predicir el valor de una variable y en base a otra variable x

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Importando Datos
temp = pd.read_csv("1.csv")

plt.figure(figsize=(8, 6))

# Usar sns.scatterplot() con ambas columnas como argumentos
sns.scatterplot(x='Celsius', y='Fahrenheit', data=temp)

# Agregar títulos y etiquetas de los ejes
plt.title('Relación entre Celsius y Fahrenheit')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')

# Mostrar el gráfico
#plt.show()


#Cargar set de datos
X_train =  temp['Celsius']
y_train =  temp['Fahrenheit']

#Se va a crear un modelo de forma secuencial
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#aqui veremos cuantas capas esta trabajando el modelo
#model.summary()

#compilado
model.compile(optimizer = tf.keras.optimizers.Adam(1.2), loss='mean_squared_error')


#Entrenar el modelo
epochs_hist = model.fit(X_train, y_train, epochs = 100)

#Evaluando el modelo
epochs_hist.history.keys()


#graficos

'''
plt.figure(figsize=(8, 6))
plt.plot(epochs_hist.history['loss'])
plt.title('Pérdida del Modelo durante el Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.show()
'''


#predicciones

Temp_C = np.array([-50])
Temp_F = model.predict([Temp_C])

print(Temp_F)


