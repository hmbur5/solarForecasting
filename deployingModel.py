from datetime import datetime
from datetime import timedelta
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed, Dropout
from keras import Input
from keras import Model
from keras.layers import Concatenate
from keras.initializers import Orthogonal
from WorldWeatherPy import HistoricalLocationWeather
import vincent
import pandas as pd
from socket import *
import time

# weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
# use api to download historical weather data from the same place and time
keys = ['f2f090e1b01d4d7ea1435335211404', 'd6c47801209e43a6b2150339211105', 'd901813a0ad147ca829234002210505', 'a4e8b13df3d34c208ea22145210605']
api_key = keys[0]


# define model
encoder_input = Input(shape=(1,301))
forecast_input = Input(shape=(24,8))
encoder_layer_1 = LSTM(600, activation='relu', kernel_initializer=Orthogonal())
encoder_hidden_output = encoder_layer_1(encoder_input)
decoder_input = RepeatVector(24)(encoder_hidden_output)
decoder_input = Dropout(0.2)(decoder_input)
decoder_input = Concatenate(axis=2)([decoder_input, forecast_input])
decoder_layer = LSTM(600, activation='relu', return_sequences=True, kernel_initializer=Orthogonal())
decoder_output = decoder_layer(decoder_input)
dense_input = Dropout(0.2)(decoder_output)
dense_layer = TimeDistributed(Dense(300, activation='relu'))
dense_output = dense_layer(dense_input)
outputs = TimeDistributed(Dense(1))(dense_output)
model = Model(inputs=[encoder_input, forecast_input], outputs=outputs, name="model")

# Loads the trained model weights
model.load_weights('training/cp.ckpt')


customer_locations = np.load('data/customer_locations.npy',allow_pickle='TRUE').item()
customer_capacity = np.load('data/customer_capacity.npy',allow_pickle='TRUE').item()

for customer_no in range(1, 301):


    location = customer_locations[str(customer_no)][0]
    postcode = customer_locations[str(customer_no)][1]

    # customer number and capacity is input 1
    customer_number = np.zeros((1,301))
    customer_number[0,customer_no-1] = 1
    customer_number[0,1] = customer_capacity[str(customer_no)]

    # get forecast weather
    current_day = datetime.today()
    date = current_day.strftime('%Y-%m-%d')
    next_day = current_day+timedelta(days=1)
    end_date = next_day.strftime('%Y-%m-%d')
    try:
        dataset = HistoricalLocationWeather(api_key, location, date, end_date, 1).retrieve_hist_data()
    except timeout:
        time.sleep(10)
        dataset = HistoricalLocationWeather(api_key, location, date, end_date, 1).retrieve_hist_data()

    weather_data = np.array([dataset['uvIndex'],dataset['cloudcover'], dataset['humidity'], dataset['precipMM'],dataset['pressure'],
                         dataset['tempC'], dataset['visibility']])


    vals = []

    for i in range(2):
        weather_forecast = np.empty((24,8))

        weather_data_today = weather_data[:,0+i*24:24+i*24].transpose()
        weather_forecast[0:24, 0:7] = weather_data_today
        for hour in range(24):
            # adding boolean for whether sun is up
            currentTime = datetime.strptime(str(hour)+':00', '%H:%M').time()
            sunrise = datetime.strptime(dataset['sunrise'][0], '%I:%M %p').time()
            sunset = datetime.strptime(dataset['sunset'][0], '%I:%M %p').time()
            if currentTime > sunrise and currentTime < sunset:
                weather_forecast[hour, 7] = 1
            else:
                weather_forecast[hour, 7] = 0


        input_1 = np.empty((1,1,301))
        input_1[0,:,:] = customer_number
        input_2 = np.empty((1,24,8))
        input_2[0,:,:] = weather_forecast

        # normalise
        scaler0 = np.load('training/scaler0.npy', allow_pickle='TRUE').item()
        input_1[:, :, -1] = scaler0.transform(input_1[:, :, -1])
        for i in range(input_2.shape[2]):
            scaler1 = np.load('training/scaler1'+str(i)+'.npy', allow_pickle='TRUE').item()
            input_2[:, :, i] = scaler1.transform(input_2[:, :, i])

        prediction = model.predict([input_1, input_2])

        # unnormalise
        scaler2 = np.load('training/scaler2.npy',allow_pickle='TRUE').item()
        prediction = scaler2.inverse_transform(prediction[:,:,0])


        for val in prediction[0]:
            if val>=0:
                vals.append(val)
                #vals.append(int(val*1000))
            else:
                vals.append(0)

    df = pd.DataFrame()
    df['prediction'] = vals
    current_day = datetime.today()
    current_day = current_day.replace(hour=0, minute=0)
    times = []
    for i in range(48):
        times.append(current_day+timedelta(hours=i))
    df['times'] = times
    current_hour = datetime.now().replace(microsecond=0, second=0, minute=0)
    mask = (df['times'] >= current_hour) & (df['times'] < current_hour+timedelta(hours=24))
    df = df.loc[mask]
    df = df.set_index(pd.DatetimeIndex(df['times']))


    line = vincent.Line(df['prediction'], width = 450, height=200)
    line.axis_titles(x='time', y='kwatts')
    line.to_json('predictions/'+str(customer_no)+'.json')





