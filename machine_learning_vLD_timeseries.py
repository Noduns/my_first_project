# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:16:22 2017

@author: Lauren
"""

#==============================================================================
# Import packages
#==============================================================================

import warnings
warnings.filterwarnings('ignore')

#import data packages
import numpy as np
import pandas as pd
from matplotlib import pyplot

#import data preprocessing
from sklearn.model_selection import GridSearchCV

#neural net
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History 

#==============================================================================
# Import data 
#==============================================================================
timesteps = 5
data = pd.read_csv("D:/Lauren/Documents/Projet/Data generation/train_data_timeseries0.csv", names = ["DOA", "Ecart_DOA", "Dist","Ecart_Dist","Association"], index_col=False)
values = np.zeros((data.shape[0], data.shape[1], timesteps))
for i in range(0,timesteps):
    values[:,:,i] = pd.read_csv("D:/Lauren/Documents/Projet/Data generation/train_data_timeseries"+str(i)+".csv", names = ["DOA", "Ecart_DOA", "Dist","Ecart_Dist","Association"], index_col=False)

# fix random seed for reproducibility
np.random.seed(42)


#==============================================================================
# Preparing the data
#==============================================================================
#reorganize data for time series processing

look_back = 5

def series_to_recurrent(data, n_in=look_back):
	

    agg = np.zeros((data.shape[0], data.shape[1]*data.shape[2]))
    print(agg.shape)

    for i in range(0, n_in):
        agg[:,i*data.shape[1]:(i*data.shape[1]+data.shape[1])]=data[:,:,i]


    return agg

time_series = series_to_recurrent(values,look_back)


# split into train and test sets
train_test_split = int(0.8*time_series.shape[0])
time_series_train = time_series[:train_test_split, :]
time_series_test = time_series[train_test_split:, :]

# split into input and outputs
train_X, train_y = time_series_train[:, :-1], time_series_train[:, -1]
test_X, test_y = time_series_test[:, :-1], time_series_test[:, -1]

#scaling
scale = np.max(train_X)
train_X /= scale
test_X /= scale

mean = np.std(train_X)
train_X -= mean
test_X -= mean

#reshape
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

#==============================================================================
# Creating the model
#==============================================================================
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])


# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=256, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()