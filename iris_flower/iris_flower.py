import os
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout

plot_images = False
image_path = "images"

if not os.path.exists(image_path):
    os.makedirs(image_path)

#Reading data
#https://www.kaggle.com/datasets/arshid/iris-flower-dataset
data = pd.read_csv('IRIS.csv') 

print(f"Data:\n{data}")
print(f"Dataset Length: {len(data)}")
print(f"Data Columns: {list(data.columns)}\n")

#Verifying missing values
print(f"There are NaN values?: {data.isnull().values.any()}\n")
print(f"Number of NaN values:\n{data.isnull().sum()}\n")
#print(data.info())

#Discretizing target
'''
    Iris-setosa: 0
    Iris-versicolor: 1
    Iris-virginica: 2
'''
print(data['species'].unique())
data['species_disc'] = data['species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
print(data)

#Visualizing data
colors = ['purple', 'blue', 'green']
fig1 = plt.figure()
for i in range(3): #For each class make a plot in graph
    fig1 = plt.scatter(  data['sepal_length'][data['species_disc']==i],
                            data['sepal_width'][data['species_disc']==i],
                            c=colors[i],
                            label=data['species'].unique()[i])
plt.title('Iris-Flower Graph')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.savefig(os.path.join(image_path,"2D_graph.png"), dpi=300)

fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
for i in range(3): #For each class make a plot in graph
    ax.scatter( data['sepal_length'][data['species_disc']==i], 
                data['sepal_width'][data['species_disc']==i], 
                data['petal_length'][data['species_disc']==i], 
                c=colors[i], 
                label=data['species'].unique()[i])
ax.set_title("Iris-Flower Graph")
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal_Length')
ax.legend()
plt.savefig(os.path.join(image_path,"3D_graph.png"), dpi=300)
if plot_images:
    plt.show()

#Separating target from features
y = data['species_disc'] #Target
x = data.drop(columns=['species','species_disc']) #Features

print(f"Data:\n{x.head()}\n")
print(f"Target:\n{y.head()}\n")

#Separating train, val and test data
'''
    Train Data = 60%
    Validation Data = 30%
    Test Data = 10%
'''

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.6, shuffle=True, random_state=64)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, train_size=0.75, shuffle=True, random_state=64)

print(f'Total length: {len(x)}')
print(f'Train length: {len(x_train)}')
print(f'Val length: {len(x_val)}')
print(f'Test length: {len(x_test)}')

batch_size = 10
epochs = 50
loss = 'sparse_categorical_crossentropy'
optmizer = 'adam'
metrics = ['accuracy']


model = Sequential(name="iris_flower")
#The input shape will be 4 because we have four variables (sepal/petal length and sepal/petal width)
model.add(Input(shape=(4))) 
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(3, activation = 'softmax')) #Final layer using softmax because the target is multiclass
model.compile(loss=loss, optimizer=optmizer, metrics=metrics)

model.summary()

my_callbacks = [keras.callbacks.EarlyStopping(patience=8)]

history = model.fit(x = x_train, 
                    y = y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=1,
                    validation_data=(x_val,y_val),
                    callbacks=my_callbacks)

print(history.history.keys())

#Plot loss and accuracy graph
fig3 = plt.figure()
plt.plot(history.history['loss'] , c='blue', label='Train')
plt.plot(history.history['val_loss'], c='red', label="Validation")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Graph")
plt.legend()
plt.savefig(os.path.join(image_path,"loss_graph.png"), dpi=300)

fig4 = plt.figure()
plt.plot(history.history['accuracy'] , c='blue', label='Train')
plt.plot(history.history['val_accuracy'], c='red', label="Validation")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Accuracy Graph")
plt.legend()
plt.savefig(os.path.join(image_path,"accuracy_graph.png"), dpi=300)
if plot_images:
    plt.show()

#Verifica a generalização do modelo para dados de teste
score = model.evaluate(x_test, y_test, verbose=1) 
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

#Making a predicion
model_predict = model.predict(x_test)

predicts = []
for prediction in model_predict:
    predicts.append(np.argmax(prediction,axis=0))

#Plot confusion matrix
conf_matrix = confusion_matrix(y_test, predicts)
cm = pd.DataFrame(conf_matrix,
                     index = ['SETOSA','VERSICOLR','VIRGINICA'], 
                     columns = ['SETOSA','VERSICOLR','VIRGINICA'])
fig5 = plt.figure()
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig(os.path.join(image_path,"confusion_matix.png"),dpi=300)
if plot_images == True:
    plt.show()
print(conf_matrix)

model.save("iris_model.h5")
