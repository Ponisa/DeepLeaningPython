# -*- coding: utf-8 -*-

"""
# --- Author : Baruch AMOUSSOU-DJANGBAN
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Dataset 1 : Pima Indians Diabetes Database

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Laod dataset 
dataset = np.loadtxt("/home/amoussou-djangban/workspace/data/diabetes.csv", delimiter=",", skiprows=1)
X = dataset[:,0:8]
Y = dataset[:,8]

# Create model 
model = Sequential()
model.add(Dense(12,input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

# Fit the model 
model.fit(X,Y, epochs=250, batch_size=10)

# Evaluate the model
scores = model.evaluate(X,Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



