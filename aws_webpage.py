from flask import Flask
from flask import render_template
from flask import request
import folium
import pandas as pd
import folium
import pgeocode as pg
import csv
import numpy as np
import json
import requests
from datetime import datetime
from flask_apscheduler import APScheduler
import apscheduler
from folium_jsbutton import JsButton
from folium.plugins import MarkerCluster
import pickle
import random


def updatePredictions():
    print('updating')
    import deployingModel


#pgeocode global control, setting AU database
nomi = pg.Nominatim('AU')


app = Flask(__name__)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


@app.route("/")
def index():

    # in background, we update predictions every hour
    try:
        app.apscheduler.add_job(func=updatePredictions, id='j', trigger='cron', minute=0)
    # if the job is already running (happens when page is refreshed after job is started)
    except apscheduler.jobstores.base.ConflictingIdError:
        pass

    start_coords = (46.9540700, 142.7360300)
    # creating the initial map
    folium_map = folium.Map(location=[-33.3438627, 151.4919677], tiles="cartodbpositron",
                   zoom_start=8)  # zoom 6 displays all of nsw centred on sydney, might want to be closer for detail view
    fg = folium.FeatureGroup(name='Solar panel locations', show=False)
    folium_map.add_child(fg)
    marker_cluster = MarkerCluster(showCoverageOnHover=False).add_to(fg)



    customer_locations = np.load('data/customer_locations.npy', allow_pickle='TRUE').item()
    customer_capacity = np.load('data/customer_capacity.npy', allow_pickle='TRUE').item()
    customer_coordinates = {}

    for customer_no in range(1,301):
        # get prediction graph
        with open('predictions/'+str(customer_no)+'.json') as json_file:
            vis = json.load(json_file)

        # get coordinates from postcode
        postcode = customer_locations[str(customer_no)][1]
        locData = nomi.query_postal_code(postcode)
        lat = locData.latitude
        long = locData.longitude
        customer_coordinates[customer_no] = [lat, long]

        # other features for marker
        popup = folium.Popup(max_width=550).add_child(folium.Vega(vis, width=550, height=250))
        info = 'Generator capacity: '+str(customer_capacity[str(customer_no)]+' kWp')

        folium.Marker(
            location=[lat, long],
            icon=folium.Icon(color="blue"),
            tooltip=info,
            popup=popup,
        ).add_to(marker_cluster)

    # add webpage title
    loc = '24 Hour Power Generation Forecast for Home Solar Panels'
    title_html = '''
                 <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                 '''.format(loc)


    # regions
    fg1 = folium.FeatureGroup(name='Regions', show=False)
    folium_map.add_child(fg1)

    # colouring in
    fg2 = folium.FeatureGroup(name='Heat map', show=False)
    folium_map.add_child(fg2)

    with open('data/electorates.pkl', 'rb') as f:
        polygons = pickle.load(f)

    for i in range(len(polygons)):
        # Creating GeoJson file with relevant coordinates
        regionJson = {"type": "FeatureCollection",
                      "features": [
                          {"type": "Feature",
                           "geometry": {
                               "type": "Polygon",
                               "coordinates": [
                                   polygons[i][0]
                               ]
                           },
                           "properties": {
                               "name": 'hello',
                               "data": {"45.6kwH": "50"}
                           }
                           }
                      ]
                      }
        # end GeoJson - coordinates need to go above here

        js = folium.GeoJson(
            regionJson,  # GeoJson file to use
            name="Regions",
            style_function = lambda x: {'color': 'black', 'fillOpacity': 0},
            #tooltip=folium.GeoJsonTooltip(  # hover tooltip
            #    fields=['name', 'data'],  # variables, properties from GeoJson file
            #    aliases=['Region: ', 'Generator Capacity'],  # titles for properties
            #    localize=True),
            tooltip=str('This region has ' + str(len(polygons[i][1])) + ' solar panels'),
        )

        try:
            with open('predictions/polygon'+str(i+1)+'.json') as json_file:
                vis = json.load(json_file)
            js.add_child(folium.Popup(max_width=550).add_child(folium.Vega(vis, width=550, height=250)))
        except:
            pass
        js.add_to(fg1)

        js2 = folium.GeoJson(
            regionJson,  # GeoJson file to use
            name="Regions",
            style_function=lambda x: {'color': 'black', 'fillOpacity': random.uniform(0, 0.5), 'fillColor': 'green'},

        )
        js2.add_to(fg2)


        #
        #js.add_to(folium_map)  # adding GeoJson with popup data to the map

        #print(js)






    folium.LayerControl().add_to(folium_map)

    # info button
    JsButton(
        title='<i class="fas fa-info"></i>', function="""
        function(btn, map) {
            alert("Click on a region to view forecast of power generation across solar panels within the area. \\nTo view the forecast of individual solar panels, click the layer button in the top right corner, and check 'Solar panel locations'. \\n\\nThis prediction is based on the weather forecast and the past performance of the solar panels. \\nAssuming accurate weather forecast, the expected error is 0.063 kwatts. ");
        }
        """).add_to(folium_map)

    folium_map.get_root().html.add_child(folium.Element(title_html))


    # print(long,lat,locName)
    # print(pc)
    return folium_map._repr_html_()



if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8090", threaded=True, debug=False, use_reloader=False)
    #app.run(host="0.0.0.0", port="80", threaded=True, debug=False, use_reloader=False)