import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.metrics import Accuracy

#Reading data
data = pd.read_csv('IRIS.csv')

print(f"Data:\n{data}")
print(f"Dataset Length: {len(data)}")
print(f"Data Columns: {list(data.columns)}\n")

#Verifying missing values
print(f"There are NaN values?: {data.isnull().values.any()}\n")
print(f"Number of NaN values:\n{data.isnull().sum()}\n")

#Discretizing target
'''
    Iris-setosa: 0
    Iris-versicolor: 1
    Iris-virginica: 2
'''
print(data['species'].unique())
data['species'] = data['species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
print(data)

#Separating target from features
y = data['species'] #Target
x = data.drop(columns='species') #Features

print(f"Data:\n{x.head()}\n")
print(f"Target:\n{y.head()}\n")


#Separating train, val and test data
'''
    Train Data = 60%
    Validation Data = 30%
    Test Data = 10%
'''

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.6, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, train_size=0.75, shuffle=True)

print(f'Total length: {len(x)}')
print(f'Train length: {len(x_train)}')
print(f'Val length: {len(x_val)}')
print(f'Test length: {len(x_test)}')

batch_size = 64
epochs = 35
loss = 'sparse_categorical_crossentropy'
optmizer = 'adam'

model = Sequential(name="iris_flower")
#The input shape will be 4 because we have four variables (sepal/petal length and sepal/petal width)
model.add(Input(shape=(4))) 
model.add(Dense(32, activation = 'tanh'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(3, activation = 'softmax')) #Final layer using softax because the target is multiclass
model.add(Flatten())
model.compile(loss=loss, optimizer=optmizer, metrics='accuracy')

model.summary()

history = model.fit(   x = x_train, 
                    y = y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=1,
                    validation_data=(x_val,y_val))

#Verifica a generalização do modelo para dados de teste
model.evaluate(x_test, y_test, verbose=1) 

print(history.history.keys())

# model.predict(x_test)
# model.save("Path.h5")
