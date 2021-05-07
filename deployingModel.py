import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed, Dropout, MaxPooling1D, AveragePooling1D
from keras import regularizers
from keras import Input
from keras import Model
from keras.layers import Concatenate
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from keras import models
import pickle
import random
from math import floor, ceil
from keras.initializers import Orthogonal
import os
from WorldWeatherPy import HistoricalLocationWeather

# weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
# use api to download historical weather data from the same place and time
api_key = 'a4e8b13df3d34c208ea22145210605'


# define model
encoder_input = Input(shape=(1,1))
forecast_input = Input(shape=(24,8))
encoder_layer_1 = LSTM(20, activation='relu', kernel_initializer=Orthogonal())
encoder_hidden_output = encoder_layer_1(encoder_input)
decoder_input = RepeatVector(24)(encoder_hidden_output)
decoder_input = Dropout(0.2)(decoder_input)
decoder_input = Concatenate(axis=2)([decoder_input, forecast_input])
decoder_layer = LSTM(20, activation='relu', return_sequences=True, kernel_initializer=Orthogonal())
decoder_output = decoder_layer(decoder_input)
dense_input = Dropout(0.2)(decoder_output)
dense_layer = TimeDistributed(Dense(100, activation='relu'))
dense_output = dense_layer(dense_input)
outputs = TimeDistributed(Dense(1))(dense_output)
model = Model(inputs=[encoder_input, forecast_input], outputs=outputs, name="model")

# Loads the trained model weights
model.load_weights('training/cp.ckpt')



# testing on a location
customer_locations = np.load('data/customer_locations.npy',allow_pickle='TRUE').item()
customer_no = 1
location = customer_locations[str(customer_no)]

# customer number is input 1
customer_number = np.empty((1,1))
customer_number[0,0] = customer_no

# get forecast weather
weather_forecast = np.empty((24,8))
current_day = datetime.today()
date = current_day.strftime('%Y-%m-%d')
next_day = current_day+timedelta(days=1)
end_date = next_day.strftime('%Y-%m-%d')
city = location+'+Australia'
dataset = HistoricalLocationWeather(api_key, city, date, end_date, 1).retrieve_hist_data()
weather_data = np.array([dataset['uvIndex'],dataset['cloudcover'], dataset['humidity'], dataset['precipMM'],dataset['pressure'],
                     dataset['tempC'], dataset['visibility']])
weather_data = weather_data[:,0:24].transpose()
weather_forecast[0:24, 0:7] = weather_data
for hour in range(24):
    # adding boolean for whether sun is up
    currentTime = datetime.strptime(str(hour)+':00', '%H:%M').time()
    sunrise = datetime.strptime(dataset['sunrise'][0], '%I:%M %p').time()
    sunset = datetime.strptime(dataset['sunset'][0], '%I:%M %p').time()
    if currentTime > sunrise and currentTime < sunset:
        weather_forecast[hour, 7] = 1
    else:
        weather_forecast[hour, 7] = 0



input_1 = np.empty((1,1,1))
input_1[0,:,:] = customer_number
input_2 = np.empty((1,24,8))
input_2[0,:,:] = weather_forecast

# normalise
for i in range(input_2.shape[2]):
    scaler1 = np.load('training/scaler1'+str(i)+'.npy', allow_pickle='TRUE').item()
    input_2[:, :, i] = scaler1.transform(input_2[:, :, i])

prediction = model.predict([input_1, input_2])

# unnormalise
scaler2 = np.load('training/scaler2.npy',allow_pickle='TRUE').item()
prediction = scaler2.inverse_transform(prediction[:,:,0])

plt.plot(prediction[0])
plt.show()