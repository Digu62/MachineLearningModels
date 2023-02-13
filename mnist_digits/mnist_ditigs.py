import os
import numpy as np
import pandas as pd
import seaborn as sb
from random import randint
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten

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

#Resahping to plot images
resheaped = np.array(x_train).reshape(-1,28,28)
print(f"Reshaped:\n {resheaped.shape}")
plt.figure(figsize=(7,7))
for i in range(9):
    plt.subplot(3,3, i+1)
    image = randint(0,len(x_train))
    plt.imshow(resheaped[image],cmap='gray',interpolation='none')
    plt.title(f"Label {y_train[image]}")
    plt.axis('off')
plt.show()
