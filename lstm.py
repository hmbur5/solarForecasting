import csv
import random
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed, Dropout
from keras import Input
from keras import Model
from keras.layers import Concatenate
from keras.callbacks import ModelCheckpoint
import random
from keras.initializers import Orthogonal
import os

# get all csv files
directory = 'data/'
locations = []
for filename in os.listdir(directory):
    if filename.endswith("weather.csv"):
        filename = filename.replace('simplified ', '')
        filename = filename.replace(' weather.csv', '')
        try:
            int(filename)
        except:
            continue
        else:
            locations.append(int(filename))
    else:
        continue
customer_capacity = np.load('data/customer_capacity.npy',allow_pickle='TRUE').item()

first_input_sequences = np.empty((0,1,2))
second_input_sequences = np.empty((0,24,8))
output_sequences = np.empty((0,24,1))
days_sequences = []

for location in locations:

    # splitting data up into days so we can simply predict the solar generation on a given day
    days = []
    with open("data/simplified "+str(location)+" solar.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
            if day not in days:
                days.append(day)

    # get input from weather data
    input_sequence = np.empty((len(days),24,8), dtype=np.float64)
    with open("data/simplified "+str(location)+" weather.csv", 'r') as csvfile:
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
    with open("data/simplified "+str(location)+" solar.csv", 'r') as csvfile:
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

    # first input sequence is the customer index and solar panel capacity. add for each day
    customer_number = np.empty((1,1,2))
    customer_number[0,0,0] = location
    customer_number[0,0,1] = customer_capacity[str(location)]
    for day in days:
        first_input_sequences = np.append(first_input_sequences, customer_number, axis=0)

    input_sequence = np.array(input_sequence)
    output_sequence = np.array(output_sequence)
    days = np.array(days)


    second_input_sequences = np.append(second_input_sequences, input_sequence, axis=0)
    output_sequences = np.append(output_sequences, output_sequence, axis=0)
    day_customer = []
    for day in days:
        day_customer.append(str(day)+' Customer '+ str(location))
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


# get training data to use for normalising
training_portion = np.float64(0.8)
train_size = int(len(days_sequences) * training_portion)
train_input_historic = first_input_sequences[0: train_size]
train_input_weather = second_input_sequences[0: train_size]
train_output = output_sequences[0: train_size]

# normalise based on training data, but apply to all data
scaler0s = [MinMaxScaler(feature_range=(0, 1))]*first_input_sequences.shape[2]
for i in range(first_input_sequences.shape[2]):
    scaler0 = scaler0s[i]
    scaler0 = scaler0.fit(train_input_historic[:, :, i])
    scaler0s[i] = scaler0
    first_input_sequences[:, :, i] = scaler0.transform(first_input_sequences[:, :, i])
    # save scalers to file
    np.save('training/scaler0'+str(i)+'.npy', scaler0)
scaler1s = [MinMaxScaler(feature_range=(0, 1))]*second_input_sequences.shape[2]
for i in range(second_input_sequences.shape[2]):
    scaler1 = scaler1s[i]
    scaler1 = scaler1.fit(train_input_weather[:, :, i])
    scaler1s[i] = scaler1
    second_input_sequences[:, :, i] = scaler1.transform(second_input_sequences[:, :, i])
    # save scalers to file
    np.save('training/scaler1'+str(i)+'.npy', scaler1)
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler2 = scaler2.fit(train_output[:, :, 0])
output_sequences[:, :, 0] = scaler2.transform(output_sequences[:, :, 0])

np.save('training/scaler2.npy', scaler2)

# split days into training and testing
training_portion = np.float64(0.8)
train_size = int(len(days_sequences) * training_portion)
train_input_historic = first_input_sequences[0: train_size]
train_input_weather = second_input_sequences[0: train_size]
train_output = output_sequences[0: train_size]
validation_input_historic = first_input_sequences[train_size:-6]
validation_input_weather = second_input_sequences[train_size:-6]
validation_output = output_sequences[train_size:-6]
test_input_historic = first_input_sequences[-6:]
test_input_weather = second_input_sequences[-6:]
test_output = output_sequences[-6:]
test_days = days_sequences[-6:]

print(np.shape(train_input_historic))

# create callback to save model weights each time its trained (for deployment)
cp_callback = ModelCheckpoint(filepath='training/cp.ckpt',
                                                save_best_only=True,
                                                 verbose=1)

# define model
encoder_input = Input(shape=(1,2))
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


model.compile(loss='mse', optimizer='adam')
num_epochs = 100
model.fit([train_input_historic, train_input_weather], train_output, epochs=num_epochs, validation_data=
    ([validation_input_historic, validation_input_weather], validation_output), verbose=2, callbacks=[cp_callback])


# using testing data to plot some days
prediction = model.predict([test_input_historic, test_input_weather])
observation = test_output

# unnormalise
prediction = scaler2.inverse_transform(prediction[:,:,0])
observation = scaler2.inverse_transform(observation[:,:,0])


max = max([np.max(prediction), np.max(observation)])
min = min([np.min(prediction), np.min(observation)])
fig, axes = plt.subplots(2,3)
for i,j in [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2]]:
    axes[i,j].plot(observation[i*2+j])
    axes[i,j].plot(prediction[i*2+j])
    axes[i,j].set_title(str(test_days[i*2+j]))
    axes[i, j].set_ylabel('kwatt hours')
    try:
        axes[i,j].set_ylim([min,max])
    except ValueError:
        pass
axes[0,0].legend(['observation', 'prediction'])
plt.show()