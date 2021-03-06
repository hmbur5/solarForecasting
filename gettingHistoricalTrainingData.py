import csv
import pandas as pd
import datetime
import pgeocode as pg
import numpy as np
from WorldWeatherPy import HistoricalLocationWeather
# weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
# use api to download historical weather data from the same place and time
keys = ['f2f090e1b01d4d7ea1435335211404', 'd6c47801209e43a6b2150339211105', 'd901813a0ad147ca829234002210505', 'a4e8b13df3d34c208ea22145210605']
api_key = keys[0]

#pgeocode global control, setting AU database
nomi = pg.Nominatim('AU')

# solar data downloaded from https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
# opening solar data file, and getting data for customers
customerDict = {}
# load customer locations dictionary, with first element place name, second element postcode
try:
    customer_locations = np.load('data/customer_locations.npy',allow_pickle='TRUE').item()
except Exception as e:
    print(e)
    customer_locations = {}

customer_capacity = {}

with open('data/2012-2013 Solar home electricity data v2.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader:
        if '2012-2013 Solar' in lines[0]:
            continue
        if lines[0] == 'Customer':
            header = lines
        else:
            # only using generation data
            if lines[3] == 'GG':
                try:
                    customerDict[lines[0]].append(lines)
                except KeyError:
                    customerDict[lines[0]] = [lines]
            try:
                customer_capacity[lines[0]]
            except KeyError:
                customer_capacity[lines[0]]= lines[1]
                print(lines[2])
                customer_locations[lines[0]] = (customer_locations[lines[0]], lines[2])
                np.save('data/customer_capacity.npy', customer_capacity)
                np.save('data/customer_locations.npy', customer_locations)

for customer in customerDict.keys():
    customerList = customerDict[customer]

    # convert to dataframe
    customerData = pd.DataFrame(data=customerList, columns=header)
    #print(customerList[1][2]) change this to pandas data frame, was having issues so used this as proof of concept

    #Reading postcode data, creating data frame and isolating location name
    #locPostcode = customerdata.loc[1,"Postcode"] # semihardcoded for customer 1, will need to introduce variation - see above
    locPostcode = customerList[1][2] #not the best method, see print line comment
    locData = nomi.query_postal_code(locPostcode)
    locName = locData.place_name
    state = locData.state_name


    # splitting in case of multiple locations in same postcode
    chunks = locName.split(', ')
    done=False
    for chunk in chunks:
        locName = chunk
        # use api to download historical weather data from the same place and time
        city = locName.replace(' ', '+')+'+'+state.replace(' ','+')+'+Australia'
        start_date = '2012-07-01'
        end_date = '2013-6-30'
        frequency = 1  # hourly frequency
        # skip over locations that have already been done
        try:
            with open('data/simplified ' + customer + ' weather.csv', "r") as csv_file:
                done=True
                break
        except FileNotFoundError:
            pass
        try:
            dataset = HistoricalLocationWeather(api_key, city, start_date, end_date, frequency).retrieve_hist_data()
        except KeyError:
            print(locName)
        else:
            break

    if done==True:
        continue

    print(locName)
    customer_locations[customer] = city


    # convert generation data to hourly rather than half hourly (so it matches with weather)
    for i in range(24):
        half_hour = str(i)+':30'
        if i+1==24:
            hour = '0:00'
        else:
            hour = str(i+1)+':00'
        customerData[half_hour] = pd.to_numeric(customerData[half_hour],downcast="float")
        customerData[hour] = pd.to_numeric(customerData[hour],downcast="float")
        customerData[hour] = customerData[half_hour] + customerData[hour]
        del customerData[half_hour]

    # convert to rows corresponding to hours rather than columns (so it matches with weather)
    row_data = pd.DataFrame([], columns = ['date_time', 'power generation kwh'])
    for index, row in customerData.iterrows():
        # iterating through hours (starting at 1, then to midnight
        for i in list(range(1,24))+[0]:
            hour = str(i) + ':00'
            date = customerData.at[index, 'date']
            date_time_obj = datetime.datetime.strptime(date+' '+hour, '%d/%m/%Y %H:%M')
            # if it gets to midnight, switch to next day (based on convention used in weather data)
            if hour == '0:00':
                date_time_obj = date_time_obj+datetime.timedelta(days=1)
            date_time_str = date_time_obj.strftime('%Y-%m-%d %H:%M:%S')

            generation = customerData.at[index, hour]

            appending_dataframe = pd.DataFrame([[date_time_str, generation]], columns = ['date_time', 'power generation kwh'])

            row_data = row_data.append(appending_dataframe)


    # save this data
    row_data.to_csv('data/simplified '+customer+' solar.csv',index=False)









    # deleting some data for simplicity, particularly stuff that doesn't appear in forecasting
    del dataset['maxtempC']
    del dataset['mintempC']
    del dataset['totalSnow_cm']
    del dataset['sunHour']
    del dataset['moon_illumination']
    del dataset['moonrise']
    del dataset['moonset']
    del dataset['DewPointC']
    del dataset['FeelsLikeC']
    del dataset['HeatIndexC']
    del dataset['WindChillC']
    del dataset['WindGustKmph']

    # save data
    dataset.to_csv('data/simplified '+ customer +' weather.csv')

    # save customer locations to file
    np.save('data/customer_locations.npy', customer_locations)
