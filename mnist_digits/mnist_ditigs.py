import os
import numpy as np
import pandas as pd
import seaborn as sb
from random import randint
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

import mlflow #https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html

image_path = 'images'

#Cria uma pasta para armazenar as imagens
if not os.path.exists(image_path):
    os.makedirs(image_path)

#CHECK THIS
# #Read the wine-quality csv file from the URL
# csv_url = (
#     "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/data/winequality-red.csv"
# )
# try:
#     data = pd.read_csv(csv_url, sep=";")
# except Exception as e:
#     logger.exception(
#         "Unable to download training & test CSV, check your internet connection. Error: %s", e
#     )

data = pd.read_csv("train.csv")

print(f"Data:\n {data}")
print(f"Columns:\n {data.columns}")

#Verifying missing values
print(f"NaN Values?\n {data.isnull().values.any()}")

x_train = data.drop(['label'],axis=1)
y_train = data['label']
print(f"Shape:\n {x_train.shape}")

#Converte o array para o formato de imagem
x_train = x_train.values.reshape((x_train.shape[0],28,28,1))

#Normaliza os dados
x_train = x_train/255

#Resahping to plot images
# resheaped = np.array(x_train).reshape(-1,28,28)
print(f"Reshaped:\n {x_train.shape}")
plt.figure(figsize=(7,7))
for i in range(9):
    plt.subplot(3,3, i+1)
    image = randint(0,len(x_train))
    plt.imshow(x_train[image],cmap='gray',interpolation='none')
    plt.title(f"Label {y_train[image]}")
    plt.axis('off')
plt.show()

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.2, random_state=42)


print(f"Train shape: {x_train.shape}")
print(f"Train y shape: {y_train.shape}")
print(f"Val shape: {x_val.shape}")
print(f"Val y shape: {y_val.shape}")

#Creating model
model = Sequential(name="Mnist_digits")
model.add(Input(shape=(28,28,1), name="input_layer"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D((2,2,)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 64
epochs = 50

history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val,y_val))


