import csv
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
from datetime import datetime



# splitting the data up into days so we can simply predict the solar generation on a given day
days = []
with open("data/simplified Wahroonga solar.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
        if day not in days:
            days.append(day)

# get input from weather data
input_sequence = np.empty((len(days),24,8), dtype=np.float64)
with open("data/simplified Wahroonga weather.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        current_day = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S').date()
        index = days.index(current_day)
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
with open("data/simplified Wahroonga solar.csv", 'r') as csvfile:
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


# normalise (should only normalise in terms of the train data, but leaving for now)
scaler = MinMaxScaler(feature_range=(0, 1))
for i in range(input_sequence.shape[1]):
    scaler = scaler.fit(input_sequence[:, i, :])
    input_sequence[:, i, :] = scaler.transform(input_sequence[:, i, :])
scaler2 = MinMaxScaler(feature_range=(0, 1))
for i in range(output_sequence.shape[1]):
    scaler2 = scaler2.fit(output_sequence[:, i, :])
    output_sequence[:, i, :] = scaler2.transform(output_sequence[:, i, :])



# split days into training and testing
training_portion = np.float64(0.8)
train_size = int(len(days) * training_portion)
train_input = input_sequence[0: train_size]
train_output = output_sequence[0: train_size]
validation_input = input_sequence[train_size:-4]
validation_output = output_sequence[train_size:-4]
test_input = input_sequence[-4:]
test_output = output_sequence[-4:]
test_days = days[-4:]



# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(200, input_shape=(train_input.shape[1],train_input.shape[2]), return_sequences=True))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(train_output.shape[2]))

model.compile(loss='mse', optimizer='adam')
num_epochs = 100
history = model.fit(train_input, train_output, epochs=num_epochs, validation_data=(validation_input, validation_output), verbose=2)



# using testing data to plot some days
prediction = model.predict(test_input)
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
