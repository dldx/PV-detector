# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.1'
#   kernelspec:
#     display_name: Python [conda env:PV_detection]
#     language: python
#     name: conda-env-PV_detection-py
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.6
# ---

# + {}
import numpy as np
import pandas as pd

import geopandas as gpd

# %matplotlib inline
# -

# ### Get locations of solar arrays according to [Bradbury et al](https://www.nature.com/articles/sdata2016106#data-records)

# Read data from malformed geojson

def read_geojson(geojson_file):
    import json
    import fiona
    from shapely.geometry import shape

    """When supplied with a malformed geojson, this function tries to return a geodataframe"""
    collection = list(fiona.open(geojson_file,'r'))
    
    df1 = pd.DataFrame(collection)

    #Check Geometry
    def isvalid(geom):
        try:
            shape(geom)
            return 1
        except:
            return 0
    df1['isvalid'] = df1['geometry'].apply(lambda x: isvalid(x))
    df1 = df1[df1['isvalid'] == 1]
    collection = json.loads(df1.to_json(orient='records'))

    #Convert to geodataframe
    return gpd.GeoDataFrame.from_features(collection)

PV_locations = read_geojson("./Data/SolarArrayPolygons.geojson")

PV_locations.head()

# ### Load GeoTiff and plot it

# + {"scrolled": false}
def plot_geotiff(image_id):
    "Return a matplotlib plot with PVs overlayed"

    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib
    import matplotlib.pyplot as plt


    city = PV_locations[(PV_locations.image_name == image_id)].iloc[0].city
    pv_arrays_in_image = PV_locations[(PV_locations.image_name == image_id)].sort_values("area_meters", ascending=False)

    fname = './Data/{0}/{1}.tif'.format(city, image_id)

    img = matplotlib.image.imread(fname)

    fig, ax = plt.subplots(figsize=(50, 50), dpi=100)

    ax.imshow(img, origin='upper')

    patches = []

    for i, row in pv_arrays_in_image.iterrows():
        # convert coords to list of lists with eval
        polygon = Polygon(eval(row["polygon_vertices_pixels"]), True)
        patches.append(polygon)

    p = PatchCollection(patches, color="yellow", linewidth=4, alpha=0.8)

    ax.add_collection(p)

    return plt
# -

plot_geotiff("10sfg735670")

def plotly_geotiff(image_id, plot_width=1000, plot_height=1000):
    """When given an image id, show the image along with an overlay containing all visible PVs using plotly"""
    # First get which city the image belongs to
    city = PV_locations[(PV_locations.image_name == image_id)].iloc[0].city
    pv_arrays_in_image = PV_locations[(PV_locations.image_name == image_id)].sort_values("area_meters", ascending=False)

    import plotly.offline as py
    import plotly.graph_objs as go

    py.init_notebook_mode()

    # Original size of image
    img_width = 2000
    img_height = 2000
    scale_factor = 1

    layout= go.Layout(
                    xaxis = go.layout.XAxis(showticklabels = False,
                                            showgrid=False,
                                            zeroline=False,
                                            range = [0,img_width*scale_factor],
                                           ),
                    yaxis = go.layout.YAxis(showticklabels = False,
                                            showgrid=False,
                                            zeroline=False,
                                            scaleanchor = 'x',
                                            range = [0,img_height*scale_factor]
                                           ),
                    autosize=False,
                    # Size of plotly graph
                    height=plot_height,
                    width=plot_width,
        margin = {'l': 0, 'r': 0, 't': 0, 'b': 0},
                    images= [dict(
                        source="Data/{0}/{1}.jpg".format(city, image_id),
                        x= 0,
                        sizex=img_width*scale_factor,
                        y=img_height*scale_factor,
                        sizey=img_height*scale_factor,
                        xref="x", yref="y",
                        sizing= "stretch",
                        opacity= 1,
                        layer= "below")]
                     )

    PVs = []
    
    for i, row in pv_arrays_in_image.iterrows():
        polygon_pixels = np.array(eval(row["polygon_vertices_pixels"]))
        
        PVs.append(go.Scattergl(
        x=polygon_pixels[:,0] * img_width/5000,
        y=(5000-polygon_pixels[:,1])*img_height/5000,
            name = i,
            text = "Lat: {lat:.3f}° Long: {long:.3f}° Area: {area:.0f}m²".format(lat=float(row["centroid_latitude"]),
                                                                                 long=float(row["centroid_longitude"]),
                                                                                 area=row["area_meters"]),
            mode='lines',
            showlegend = True,
            hoveron = 'fills',
            fill='toself',
            line = dict(width=4, color='blue')))

    py.iplot(dict(data=PVs,layout=layout))

show_pvs("10sfg735670", 800, 800)
