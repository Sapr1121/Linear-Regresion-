#Proyecto 2
#Prediccion de ingresos
#Eres dueno de un negocio de helados y quieres crear un modelo para predecir los ingresos diarios en dolares basados en la temperatura (degC)

#importando librerias
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Imporetando los datos 
dataSets = pd.read_csv("2.csv")


plt.figure(figsize=(8, 6))

# Usar sns.scatterplot() con ambas columnas como argumentos
sns.scatterplot(x='Temperature', y='Revenue', data=dataSets)

# Agregar títulos y etiquetas de los ejes
plt.title('Relación entre Temperature y Revenue')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
#plt.show()


#Cargar sets de datos
X_train = dataSets['Temperature']
Y_train = dataSets['Revenue']


#Se va a crear un modelo de forma secuencial
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))


#Compilado
model.compile(optimizer = tf.keras.optimizers.Adam(0.2), loss='mean_squared_error')


#Entrenar el modelo
epochs_hist = model.fit(X_train, Y_train, epochs = 1000)

#Evaluando el modelo
epochs_hist.history.keys()

#Graficarlo
plt.figure(figsize=(8, 6))
plt.plot(epochs_hist.history['loss'])
plt.title('Pérdida del Modelo durante el Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
#plt.show()

#prediccion

Temperatura = float(input("Ingrese la temperatura: "))
TemperaturaArray = np.array([[Temperatura]])
Revenue = model.predict([TemperaturaArray])

print(Revenue)

#Graficamos la prediccion
plt.figure(figsize=(8, 6))
plt.scatter(X_train, Y_train, color = 'gray')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.show()