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

# Evaluate performance of deep learning models

# --- Automatoc validation
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, validation_split=0.33, epochs=150, batch_size=10)

# Manual validation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=seed)

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, validation_data=(X_test,y_test), epochs=150, batch_size=10)

# k-fold cross validation 
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10 , shuffle=True, random_state=seed)
cvscores=[]

for train, test in kfold.split(X,Y):
    # create model
    model = Sequential()
    model.add(Dense(12,input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    # Compile model 
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    # Fit model 
    model.fit(X[train],Y[train], epochs=150, batch_size=10)
    
    # Evaluate model
    scores = model.evaluate(X[test],Y[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))    




