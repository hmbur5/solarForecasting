from geopandas import GeoDataFrame
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import pickle


# returns polygon coordinates for electorates, and which customers are contained within these coordinates
def getRegions(customer_coordinates):
    df = GeoDataFrame.from_file("data/NSW_Electoral_Boundaries_25-02-2016.MIF")


    polygons = []
    count = 0
    for polygon in df['geometry']:
        contained = []
        for key in customer_coordinates.keys():
            lat,long = customer_coordinates[key]
            coord = Point(long, lat)
            if coord.within(polygon):
                contained.append(key)
        try:
            coords_list = list(polygon.exterior.coords)
        except AttributeError:
            for little_polygon in polygon:
                coords_list = list(little_polygon.exterior.coords)
                count+=1
                new_coords_list = []
                for element in coords_list:
                    new_coords_list.append([element[0], element[1]])
                polygons.append([new_coords_list, contained])
                continue
        count+=1
        new_coords_list = []
        for element in coords_list:
            new_coords_list.append([element[0],element[1]])
        polygons.append([new_coords_list, contained])

    with open('data/electorates.pkl', 'wb') as f:
        pickle.dump(polygons, f)

    return polygons

