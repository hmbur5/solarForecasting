# to do
# find way to turn postcodes into location strings (API?) - done
# implement locName variable in relevant locations - done
# selecting the correct location name from postcodes with multiple names - done, could be better
# concatonate into single file
# pulling postcodes from large data sets
# expanding to be able to do multiple customers
# 

## Scaling up:
# input: solar data from gov
# desired outputs:
#   single file with all customer data
#   every customer listed going vertically, maybe geogroup later
#   needs to include werather date loop to keep postcode constant


import csv
import pandas as pd
import datetime

## Using pgeocode to translate postcodes to name strings:
# nomi = pg.Nominatim('AU')
# frame = nomi.query_postal_code("3168")
# namestring = frame.place_name  
#
# installing: pip install pgeocode
import pgeocode as pg 

# weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
# use api to download historical weather data from the same place and time
from WorldWeatherPy import HistoricalLocationWeather
api_key = 'f2f090e1b01d4d7ea1435335211404'

#pgeocode global control, setting AU database
nomi = pg.Nominatim('AU')

#looping code
n = 5   #total number of customers to compute up to (could use total length in final implementation)
last_postcode = 0   #flag to trigger if postcode changes, meaning weather api needs to be called

for i in range(n):
    # solar data downloaded from https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
    # opening solar data file, and getting data for customer 1
    with open('data\\2012-2013 Solar home electricity data v2.csv', "r") as csv_file: #change back to: data\2012-2013 Solar home electricity data v2.csv
        csv_reader = csv.reader(csv_file, delimiter=',')
        customer = []
        for lines in csv_reader:
            # limit to first customer
            if lines[0] == 'Customer':
                header = lines
            elif lines[0] < str(n): 
                # just add generation data
                if lines[3]=='GG':
                    customer.append(lines)
        
    # convert to dataframe
    customerData = pd.DataFrame(data=customer, columns=header)
    #print(customer[1][2]) change this to pandas data frame, was having issues so used this as proof of concept

    #Reading postcode data, creating data frame and isolating location name
    #locPostcode = customerdata.loc[1,"Postcode"] # semihardcoded for customer 1, will need to introduce variation - see above
    locPostcode = customer[1][2] #not the best method, see print line comment
    locData = nomi.query_postal_code(locPostcode) 
    locName = locData.place_name
    print(locName)

    # splitting in case of multiple locations in same postcode
    chunks = locName.split(',')
    locName = chunks[0]
    print(locName)


    # weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
    # use api to download historical weather data from the same place and time
    start_date = '2012-07-01' #scale this to pull from data eventually
    end_date = '2013-6-30'
    city = locName
    frequency = 1 # hourly frequency
    if locPostcode != last_postcode:    #only calling the api every time the location changes, allows for using the same document for all data
        dataset = HistoricalLocationWeather(api_key, city, start_date, end_date, frequency).retrieve_hist_data()

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

        # save weather data to csv
        dataset.to_csv('data\simplified n5 weather.csv', mode='a') #Appending rather than writing in order to keep continously adding from the loop (may cause issues if not file created?)
        #including writing only when changed due to appending

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

            dataset_gen = pd.DataFrame([[generation]], columns = ['power generation kwh']) #creating dataframe of generation data

            dataset.append(dataset_gen) #including the power generation data attached to the weather dataset

    # save this data
    row_data.to_csv('data\simplified n5 solar.csv',index=False,mode='a') #change back to: data\simplified Wahroonga solar.csv
    # save weather data and generation to crossover csv
    dataset.to_csv('data\simplified n5 hybrid.csv',mode='a') #swap back to: data\simplified Wahroonga weather.csv

#end of loop, all program functions included in loop
# -writing all appending into documents as progressing through loop
# -updating weather data whenever a new postcode is detected
# -weather data is then added to the file only when changed, resulting in (n*data) weather output
# -solar data will loop through entire doucment and stop when customer (n) is reached

