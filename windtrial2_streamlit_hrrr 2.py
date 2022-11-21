# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:35:25 2022

@author: mritchey
"""
# streamlit run "C:\Users\mritchey\.spyder-py3\Python Scripts\streamlit projects\hrrr\windtrial2_streamlit_hrrr 2.py"
import base64
import datetime
import itertools
import os

import branca.colormap as cm
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import rioxarray
import s3fs

import streamlit as st
import xarray as xr
from geogif import gif
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from matplotlib import colors as colors
from pyproj import CRS, Transformer
from streamlit_folium import st_folium

os.environ['PROJ_LIB'] = r"C:\Users\mritchey\Downloads\WPy64-31040\python-3.10.4.amd64\Lib\site-packages\osgeo\data\proj"

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=0).encode('utf-8')

@st.cache
def geocode(address):
    try:
        address2 = address.replace(' ', '+').replace(',', '%2C')
        df = pd.read_json(
            f'https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?address={address2}&benchmark=2020&format=json')
        results = df.iloc[:1, 0][0][0]['coordinates']
        lat, lon = results['y'], results['x']
    except:
        geolocator = Nominatim(user_agent="GTA Lookup")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geolocator.geocode(address)
        lat, lon = location.latitude, location.longitude
    return lat, lon

@st.cache
def get_data_date_time(date_time, lat, lon):
    date, time = date_time
    path = f"hrrrzarr/sfc/{date}/{date}_{time[-2:]}z_{analysis_forecast}.zarr/{vertical_level}/{variable}"
    try:
        ds = xr.open_mfdataset([lookup(path), lookup(f"{path}/{vertical_level}")],
                               engine="zarr")
        result = ds.sel(projection_x_coordinate=lon, projection_y_coordinate=lat, method="nearest")[
            variable].values*2.23694
    except:
        result = None
    date2 = pd.to_datetime(str(date)+str(time), format='%Y%m%d%H')
    df = pd.DataFrame({'Date': date2, 'MPH': result}, index=[0])
    return df

def mapvalue2color(value, cmap):

    if np.isnan(value):
        return (1, 0, 0, 0)
    else:
        return colors.to_rgba(cmap(value), 0.7)


def lookup(path):
    return s3fs.S3Map(path, s3=s3)



s3 = s3fs.S3FileSystem(anon=True)

st.set_page_config(layout="wide")
col1, col2 = st.columns((2))

address = st.sidebar.text_input(
    "Address", "123 Main Street, Columbus, OH 43215")
d = st.sidebar.date_input(
    "Date",  pd.Timestamp(2022, 9, 28)).strftime('%Y%m%d')

time = st.sidebar.selectbox('Time:', ('12 AM', '6 AM', '12 PM', '6 PM',))
type_wind = st.sidebar.selectbox('Type:', ('Gust', 'Wind'))
analysis_forecast1 = st.sidebar.radio(
    'Analysis or Forecast:', ('Analysis', 'Forecast'))
entire_day = st.sidebar.radio(
    'Graph Entire Day/Forecast (Takes a Bit):', ('No', 'Yes'))
animate_forecast = st.sidebar.radio(
    'Animate Forecast:', ('No', 'Yes'))


if analysis_forecast1 == 'Analysis':
    analysis_forecast = 'anl'
else:
    analysis_forecast = 'fcst'


if time[-2:] == 'PM' and int(time[:2].strip()) < 12:
    t = datetime.time(int(time[:2].strip())+12, 00).strftime('%H')+'00'
elif time[-2:] == 'AM' and int(time[:2].strip()) == 12:
    t = '0000'
else:
    t = datetime.time(int(time[:2].strip()), 00).strftime('%H')+'00'

year, month, day = d[:4], d[4:6], d[6:8]

if type_wind == 'Gust':
    variable = type_wind.upper()
    vertical_level = 'surface'
else:
    vertical_level = '10m_above_ground'
    if analysis_forecast == 'anl':
        variable = 'WIND_max_fcst'
    else:
        variable = 'WIND_1hr_max_fcst'

path = f"hrrrzarr/sfc/{d}/{d}_{t[:2]}z_{analysis_forecast}.zarr/{vertical_level}/{variable}"
ds = xr.open_mfdataset([lookup(path), lookup(f"{path}/{vertical_level}")],
                       engine="zarr")

lat, lon = geocode(address)

ds = ds.rename(projection_x_coordinate="x", projection_y_coordinate="y")
crs = CRS.from_cf({"grid_mapping_name": "lambert_conformal_conic",
                   "longitude_of_central_meridian": -97.5,
                   "latitude_of_projection_origin": 38.5,
                   "standard_parallel": 38.5})
ds = ds.rio.write_crs(crs, inplace=True)

ds[variable] = ds[variable].astype("float64")
projected = ds.rio.reproject("EPSG:4326")

if analysis_forecast == 'fcst':
    projected_org = projected
    projected = projected.sel(time=projected.time.values[0])
else:
    pass

wind_mph = projected.sel(x=lon, y=lat, method="nearest")[
    variable].values*2.23694


affine = projected.rio.transform()

rows, columns = rasterio.transform.rowcol(affine, lon, lat)

size = 40


projected2 = projected[variable][rows -
                                 size:rows+size, columns-size:columns+size]
img = projected2.values*2.23694

boundary = projected2.rio.bounds()
left, bottom, right, top = boundary

img[img < 0.0] = np.nan

clat = (bottom + top)/2
clon = (left + right)/2

vmin = np.floor(np.nanmin(img))
vmax = np.ceil(np.nanmax(img))

colormap = cm.LinearColormap(
    colors=['blue', 'lightblue', 'red'], vmin=vmin, vmax=vmax)

m = folium.Map(location=[lat, lon],  zoom_start=9, height=500)

folium.Marker(
    location=[lat, lon],
    popup=f"{wind_mph.round(2)} MPH").add_to(m)

folium.raster_layers.ImageOverlay(
    image=img,
    name='Wind Speed Map',
    opacity=.8,
    bounds=[[bottom, left], [top, right]],
    colormap=lambda value: mapvalue2color(value, colormap)
).add_to(m)


folium.LayerControl().add_to(m)
colormap.caption = 'Wind Speed: MPH'
m.add_child(colormap)

with col1:
    st.title('HRRR Model')
    st.write(f"{type_wind.title()} Speed: {wind_mph.round(2)} MPH at {time} UTC")
    st_folium(m, height=500)


crs = CRS.from_cf({"grid_mapping_name": "lambert_conformal_conic",
                   "longitude_of_central_meridian": -97.5,
                   "latitude_of_projection_origin": 38.5,
                   "standard_parallel": 38.5})

proj = Transformer.from_crs(4326, crs, always_xy=True)
lon2, lat2 = proj.transform(lon, lat)


if entire_day == 'Yes':
    if analysis_forecast == 'anl':
        times = [f'0{str(i)}'[-2:] for i in range(0, 24)]
        dates_times = list(itertools.product([d], times))
        
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(get_data_date_time)(i, lat2, lon2) for i in dates_times)

        df_all = pd.concat(results)
        df_all['MPH'] = df_all['MPH'].round(2)

    else:
        result = ds[variable].sel(x=lon2, y=lat2, method="nearest")*2.23694
        df_all = result.to_dataframe()[variable].reset_index()
        df_all.columns = ['Date', 'MPH']
        df_all['MPH'] = df_all['MPH'].round(2)
        if animate_forecast == 'Yes':
            # Not working
            gif(projected_org[variable], to='ds.gif',
                date_format='%m-%d-%Y: %I%p', cmap="RdBu_r", vmax=35)
            st.download_button(
                label="Download Gif",
                data='ds.gif',
                file_name='ds.gif',
            )
            file_ = open("ds.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="ds gif" width="1200">',
                unsafe_allow_html=True,
            )

    fig = px.line(df_all, x="Date", y="MPH")
    with col2:
        st.title(f'{analysis_forecast1}')
        st.plotly_chart(fig)

        csv = convert_df(df_all)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv')

else:
    pass

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
