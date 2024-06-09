#Proyecto 3
#Prediccion de precios bienes raices

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Importar Datos
house_df = pd.read_csv('3.csv')

#Visualizacion 
plt.figure(figsize=(8, 6))

# Usar sns.scatterplot() con ambas columnas como argumentos
sns.scatterplot(x='sqft_living', y='price', data=house_df)

# Agregar títulos y etiquetas de los ejes

plt.xlabel('sqft_living')
plt.ylabel('price')

#plt.show()

#Correlacion 
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(house_df.corr(), annot=True)
#plt.show()

#Limpieza de datos
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']


X = house_df[selected_features]
Y = house_df['price']


#Escalar los datos de X

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)



#Normalizando salida
Y = Y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(Y)


#Entrenamiento
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

#Definiendo modelo

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 100, activation = 'relu', input_shape = (7, )))
model.add(tf.keras.layers.Dense(units = 100, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 100, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 1, activation = 'linear'))

model.summary()


model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

epochs_hist = model.fit(X_train, Y_train, epochs = 100, batch_size = 50, validation_split = 0.2)


#Evaluando modelo
epochs_hist.history.keys()

#Grafica

plt.figure(figsize=(8, 6))
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Pérdida del Modelo durante el Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
#plt.show()


#Prediccion
#Definir hogar para definir sus respectivas entradas = inputs
#'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement'

X_test1 = np.array([[4, 3, 1960, 5000, 1, 2000, 3000]])


scaler1 = MinMaxScaler()
X_test_scaled1 = scaler1.fit_transform(X_test1)


#Realizar prediccion de los datos
Y_predict1 = model.predict(X_test_scaled1)


#Revertir le precio porque esta en una escala Diferente

Y_predict1 = scaler.inverse_transform(Y_predict1)

print("El precio de la casa seria {}".format(Y_predict1))

