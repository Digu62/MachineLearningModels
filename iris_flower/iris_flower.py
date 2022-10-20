import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split

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


