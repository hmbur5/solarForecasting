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
api_key = 'd901813a0ad147ca829234002210505'


# define model
encoder_input = Input(shape=(72,9))
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
model.load_weights('training/cp_good1.ckpt')



# testing on a location
location = 'Normanhurst'
# get example days for location
days = []
with open("data/simplified " + location + " solar.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
        if day not in days:
            days.append(day)

# we would be given the example days and power from the user, but for now using this.
example_days = random.sample(days,3)
example_power = [[], [], []]
example_weather = np.empty((72, 8))

# getting example power data
with open("data/simplified "+location+" solar.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        current_day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
        try:
            index = example_days.index(current_day)
        except ValueError:
            continue
        example_power[index].append(np.float64(row[1]))

# getting example weather data for these days
for day_index,day in enumerate(example_days):
    date = day.strftime('%Y-%m-%d')
    next_day = day+timedelta(days=1)
    end_date = next_day.strftime('%Y-%m-%d')
    dataset = HistoricalLocationWeather(api_key, location, date, end_date, 1).retrieve_hist_data()
    weather_data = np.array([dataset['uvIndex'],dataset['cloudcover'], dataset['humidity'], dataset['precipMM'],dataset['pressure'],
                         dataset['tempC'], dataset['visibility']])
    weather_data = weather_data[:,0:24].transpose()
    example_weather[day_index:day_index+24, 0:7] = weather_data
    for hour in range(24):
        # adding boolean for whether sun is up
        currentTime = datetime.strptime(str(hour)+':00', '%H:%M').time()
        sunrise = datetime.strptime(dataset['sunrise'][0], '%I:%M %p').time()
        sunset = datetime.strptime(dataset['sunset'][0], '%I:%M %p').time()
        if currentTime > sunrise and currentTime < sunset:
            example_weather[day_index+hour, 7] = 1
        else:
            example_weather[day_index+hour, 7] = 0

# combine example weather and power data into one list
example_data = np.empty((72, 9))
example_data[:,0:8] = example_weather
example_data[:,8] = [item for sublist in example_power for item in sublist]

# get forecast weather
weather_forecast = np.empty((24,8))
current_day = datetime.today() - timedelta(75)
date = current_day.strftime('%Y-%m-%d')
next_day = current_day+timedelta(days=1)
end_date = next_day.strftime('%Y-%m-%d')
dataset = HistoricalLocationWeather(api_key, location, date, end_date, 1).retrieve_hist_data()
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

input_1 = np.empty((1,72,9))
input_1[0,:,:] = example_data
input_2 = np.empty((1,24,8))
input_2[0,:,:] = weather_forecast
prediction = model.predict([input_1, input_2])
print(prediction[0])
plt.plot(prediction[0])
#plt.plot(np.array(dataset['tempC'][0:24].values, float)/float(max(dataset['tempC'][0:24].values)))
plt.show()