import csv
import pandas as pd
import datetime

# solar data downloaded from https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
# opening solar data file, and getting data for customer 1
with open('data/2012-2013 Solar home electricity data v2.csv', "r") as csv_file:
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

# convert to dataframe
customer1Data = pd.DataFrame(data=customer1, columns=header)

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
row_data.to_csv('data/simplified Wahroonga solar.csv',index=False)





# weather data from https://www.worldweatheronline.com/developer/api/docs/historical-weather-api.aspx
# use api to download historical weather data from the same place and time
from WorldWeatherPy import HistoricalLocationWeather
api_key = 'f2f090e1b01d4d7ea1435335211404'
city = 'Wahroonga'
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
dataset.to_csv('data/simplified Wahroonga weather.csv')
