# to do
# find way to turn postcodes into location strings (API?) - done
# implement locName variable in relevant locations - done
# selecting the correct location name from postcodes with multiple names - done, could be better
# pulling postcodes from large data sets
# expanding to be able to do multiple customers

import csv
import pandas as pd
import datetime
import pgeocode as pg 
## Using pgeocode to translate postcodes to name strings:
# nomi = pg.Nominatim('AU')
# frame = nomi.query_postal_code("3168")
# namestring = frame.place_name  
#
# installing: pip install pgeocode


#pgeocode global control, setting AU database
nomi = pg.Nominatim('AU')

# solar data downloaded from https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
# opening solar data file, and getting data for customer 1
with open('Desktop\Monash\\0. 2021\\1. Sem 1\TRC4200\Group Project\Repo\solarForecasting\data\\2012-2013 Solar home electricity data v2.csv', "r") as csv_file: #change back to: data\2012-2013 Solar home electricity data v2.csv
    csv_reader = csv.reader(csv_file, delimiter=',')
    customer1 = []
    for lines in csv_reader:
        # limit to first customer
        if lines[0] == 'Customer':
            header = lines
        if lines[0]=='1':
            # just add generation data
            if lines[3]=='GG':
                customer1.append(lines)
        #will need to expand out from here for more customers

# convert to dataframe
customer1Data = pd.DataFrame(data=customer1, columns=header)
#print(customer1[1][2]) change this to pandas data frame, was having issues so used this as proof of concept

#Reading postcode data, creating data frame and isolating location name
#locPostcode = customer1data.loc[1,"Postcode"] # semihardcoded for customer 1, will need to introduce variation - see above
locPostcode = customer1[1][2] #not the best method, see print line comment
locData = nomi.query_postal_code(locPostcode) 
locName = locData.place_name
print(locName)

# splitting in case of multiple locations in same postcode
chunks = locName.split(',')
locName = chunks[0]
print(locName)

# convert generation data to hourly rather than half hourly (so it matches with weather)
for i in range(24):
    half_hour = str(i)+':30'
    if i+1==24:
        hour = '0:00'
    else:
        hour = str(i+1)+':00'
    customer1Data[half_hour] = pd.to_numeric(customer1Data[half_hour],downcast="float")
    customer1Data[hour] = pd.to_numeric(customer1Data[hour],downcast="float")
    customer1Data[hour] = customer1Data[half_hour] + customer1Data[hour]
    del customer1Data[half_hour]

# convert to rows corresponding to hours rather than columns (so it matches with weather)
row_data = pd.DataFrame([], columns = ['date_time', 'power generation kwh'])
for index, row in customer1Data.iterrows():
    # iterating through hours (starting at 1, then to midnight
    for i in list(range(1,24))+[0]:
        hour = str(i) + ':00'
        date = customer1Data.at[index, 'date']
        date_time_obj = datetime.datetime.strptime(date+' '+hour, '%d/%m/%Y %H:%M')
        # if it gets to midnight, switch to next day (based on convention used in weather data)
        if hour == '0:00':
            date_time_obj = date_time_obj+datetime.timedelta(days=1)
        date_time_str = date_time_obj.strftime('%Y-%m-%d %H:%M:%S')

        generation = customer1Data.at[index, hour]

        appending_dataframe = pd.DataFrame([[date_time_str, generation]], columns = ['date_time', 'power generation kwh'])

        row_data = row_data.append(appending_dataframe)


# save this data
row_data.to_csv('Desktop\Monash\\0. 2021\\1. Sem 1\TRC4200\Group Project\Repo\solarForecasting\data\simplified Wahroonga solar.csv',index=False) #change back to: data\simplified Wahroonga solar.csv





# weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
# use api to download historical weather data from the same place and time
from WorldWeatherPy import HistoricalLocationWeather
api_key = 'f2f090e1b01d4d7ea1435335211404'
city = locName
start_date = '2012-07-01'
end_date = '2013-6-30'
frequency = 1 # hourly frequency
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

# save data
dataset.to_csv('Desktop\Monash\\0. 2021\\1. Sem 1\TRC4200\Group Project\Repo\solarForecasting\data\simplified Wahroonga weather.csv') #swap back to: data\simplified Wahroonga weather.csv