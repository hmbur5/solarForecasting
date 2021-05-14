from flask import Flask
from flask import render_template
from flask import request
import folium
import pandas as pd
import folium
import pgeocode as pg
import csv

#pgeocode global control, setting AU database
nomi = pg.Nominatim('AU')



app = Flask(__name__)
@app.route("/")
def index():
    start_coords = (46.9540700, 142.7360300)
    # creating the initial map
    folium_map = folium.Map(location=[-33.3438627, 151.4919677], tiles="Stamen Terrain",
                   zoom_start=8)  # zoom 6 displays all of nsw centred on sydney, might want to be closer for detail view

    ##coverting postcodes to lat/long
    # reading postcodes
    pc = []  # creating empty list to kickstart loop

    # data for map use
    locData = []
    lat = []
    long = []
    locName = []
    locInfo = []

    with open('data/2012-2013 Solar home electricity data v2.csv', "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
            if '2012-2013 Solar' in lines[0]:
                continue
            if lines[0] == 'Customer':
                header = lines
            else:
                # only using generation data
                if lines[2] not in pc:  # detect when changes in postcode, only save each unique
                    pc.append(lines[2])

    # looping through the list of postcodes to create lat/long lists
    for i in range(1, len(pc)):
        locData = nomi.query_postal_code(pc[i])
        lat.append(locData.latitude)
        long.append(locData.longitude)
        locName.append(
            locData.place_name)  # might be easier to save the correct name from scraping and read from there with postcode data
        locInfo.append('Meaningful data goes in here')
        i = i + 1

    ##creating map data and displaying
    # Make a data frame with dots to show on the map - will be done by pulling from csv
    data = pd.DataFrame({
        'lon': long,
        'lat': lat,
        'name': locName,
        'info': locInfo,  # still needs to be filled with whatever we want to display
    }, dtype=str)

    # add marker one by one on the map
    for i in range(0, len(data)):
        folium.Marker(
            location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
            tooltip=data.iloc[i]['name'],
            icon=folium.Icon(color="blue"),
            popup=data.iloc[i]['info'],
        ).add_to(folium_map)
    # Show the map again

    # print(long,lat,locName)
    # print(pc)
    return folium_map._repr_html_()

if __name__ == '__main__':
    #app.run(host="127.0.0.1", port="8090", threaded=True, debug=False, use_reloader=False)
    app.run(host="0.0.0.0", port="80", threaded=True, debug=False, use_reloader=False)