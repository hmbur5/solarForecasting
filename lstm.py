import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
from datetime import datetime
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

# get all csv files
directory = 'data/'
locations = []
for filename in os.listdir(directory):
    if filename.endswith("weather.csv"):
        filename = filename.replace('simplified ', '')
        locations.append(filename.replace(' weather.csv',''))
    else:
        continue

first_input_sequences = np.empty((0,72,9))
second_input_sequences = np.empty((0,24,8))
output_sequences = np.empty((0,24,1))
days_sequences = []

for location in locations[0:2]:

    # splitting data up into days so we can simply predict the solar generation on a given day
    days = []
    with open("data/simplified "+location+" solar.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
            if day not in days:
                days.append(day)

    # get input from weather data
    input_sequence = np.empty((len(days),24,8), dtype=np.float64)
    with open("data/simplified "+location+" weather.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            current_day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
            try:
                index = days.index(current_day)
            except ValueError:
                continue
            hour = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').hour
            # using uvIndex,cloudcover,humidity,precipMM,pressure,tempC,visibility
            for index1, index2 in enumerate([1,4,5,6,7,8,9]):
                input_sequence[index, hour - 1,index1] = np.float64(row[index2])
            # adding boolean for whether sun is up
            currentTime = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').time()
            sunrise = datetime.strptime(row[2], '%I:%M %p').time()
            sunset = datetime.strptime(row[3], '%I:%M %p').time()
            if currentTime>sunrise and currentTime<sunset:
                input_sequence[index, hour - 1, 7] =1
            else:
                input_sequence[index, hour - 1, 7] = 0


    output_sequence = np.empty((len(days),24,1), dtype=np.float64)
    with open("data/simplified "+location+" solar.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            current_day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
            index = days.index(current_day)
            hour = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').hour
            output_sequence[index,hour-1,0]=np.float64(row[1])

    # skip first and last days for now as the midnight is missing from them
    input_sequence = input_sequence[1:-1]
    output_sequence = output_sequence[1:-1]
    days = days[1:-1]

    # shuffle so summer/winter data is mixed up (or validation/testing will all be same type of data)
    indices = list(range(len(days)))
    random.shuffle(indices)
    input_sequence = input_sequence[indices]
    output_sequence = output_sequence[indices]
    days = np.array(days)[indices]

    # first input sequences is an array of weather data and solar output for 50 random days
    example_data = np.empty((1,24*50, input_sequence.shape[-1]+1))
    remaining_indices = list(range(len(input_sequence)))
    example_days = random.sample(range(len(input_sequence)),50)
    # remove these 50 example days from the historic training data
    for index,day in enumerate(example_days):
        for hour in range(0,24):
            example_data[0,index*hour+hour]=np.append(input_sequence[day][hour],output_sequence[day][hour])
        remaining_indices.remove(day)
    # add 3 samples of example data as an element to first input sequence for each of the remaining days
    for example_index in range(len(remaining_indices)):
        sample_indices = random.sample(range(50),3)
        example_sample = np.empty((1,72,input_sequence.shape[-1]+1))
        for sample_index1, sample_index2 in enumerate(sample_indices):
            example_sample[0,sample_index1:sample_index1+24,:] = example_data[0,sample_index2:sample_index2+24,:]
        first_input_sequences = np.append(first_input_sequences, example_sample, axis=0)

    input_sequence = np.array(input_sequence)[remaining_indices]
    output_sequence = np.array(output_sequence)[remaining_indices]
    days = np.array(days)[remaining_indices]


    second_input_sequences = np.append(second_input_sequences, input_sequence, axis=0)
    output_sequences = np.append(output_sequences, output_sequence, axis=0)
    day_customer = []
    for day in days:
        day_customer.append(str(day)+' '+ location)
    days_sequences+=day_customer
    print(location)


# shuffle so solar panel data is mixed up (or validation/testing will all be same type of data)
indices = list(range(len(days_sequences)))
random.shuffle(indices)
first_input_sequences = first_input_sequences[indices]
second_input_sequences = second_input_sequences[indices]
output_sequences = output_sequences[indices]
days_sequences = np.array(days_sequences)[indices]
print(np.shape(first_input_sequences))
print(np.shape(second_input_sequences))
print(np.shape(output_sequences))
print(np.shape(days_sequences))


# normalise (should only normalise in terms of the train data, but leaving for now)
scaler = MinMaxScaler(feature_range=(0, 1))
for i in range(first_input_sequences.shape[1]):
    scaler = scaler.fit(first_input_sequences[:, i, :])
    first_input_sequences[:, i, :] = scaler.transform(first_input_sequences[:, i, :])
scaler1 = MinMaxScaler(feature_range=(0, 1))
for i in range(second_input_sequences.shape[1]):
    scaler1 = scaler1.fit(second_input_sequences[:, i, :])
    second_input_sequences[:, i, :] = scaler1.transform(second_input_sequences[:, i, :])
scaler2 = MinMaxScaler(feature_range=(0, 1))
for i in range(output_sequences.shape[1]):
    scaler2 = scaler2.fit(output_sequences[:, i, :])
    output_sequences[:, i, :] = scaler2.transform(output_sequences[:, i, :])



# split days into training and testing
training_portion = np.float64(0.8)
train_size = int(len(days_sequences) * training_portion)
train_input_historic = first_input_sequences[0: train_size]
train_input_weather = second_input_sequences[0: train_size]
train_output = output_sequences[0: train_size]
validation_input_historic = first_input_sequences[train_size:-4]
validation_input_weather = second_input_sequences[train_size:-4]
validation_output = output_sequences[train_size:-4]
test_input_historic = first_input_sequences[-4:]
test_input_weather = second_input_sequences[-4:]
test_output = output_sequences[-4:]
test_days = days_sequences[-4:]

print(np.shape(train_input_historic))

# create callback to save model weights each time its trained (helps stability)
cp_callback = ModelCheckpoint(filepath='training/cp.ckpt',
                                                save_best_only=True,
                                                 verbose=1)

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
print(model.summary)

# Loads the previous weights
#model.load_weights('training/cp_good1.ckpt')

model.compile(loss='mse', optimizer='adam')
num_epochs = 50
model.fit([train_input_historic, train_input_weather], train_output, epochs=num_epochs, validation_data=
    ([validation_input_historic, validation_input_weather], validation_output), verbose=2, callbacks=[cp_callback])


# using testing data to plot some days
prediction = model.predict([test_input_historic, test_input_weather])
observation = test_output


max = max([np.max(prediction), np.max(observation)])
min = min([np.min(prediction), np.min(observation)])
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.plot(observation[0])
ax1.plot(prediction[0])
ax1.set_title(str(test_days[0]))
ax1.set_ylim([min,max])
ax1.legend(['observation', 'prediction'])
ax2.plot(observation[1])
ax2.plot(prediction[1])
ax2.set_title(str(test_days[1]))
ax2.set_ylim([min,max])
ax3.plot(observation[2])
ax3.plot(prediction[2])
ax3.set_title(str(test_days[2]))
ax3.set_ylim([min,max])
ax4.plot(observation[3])
ax4.plot(prediction[3])
ax4.set_title(str(test_days[3]))
ax4.set_ylim([min,max])
plt.show()