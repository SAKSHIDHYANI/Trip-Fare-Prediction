import streamlit as st
import numpy as np
import pickle
import os
from datetime import datetime
import pandas as pd

st.set_option('deprecation.showfileUploaderEncoding',False)
d = os.path.dirname(os.getcwd())
model = d+"\\src\\models\\model_pkl"
with open(model,"rb") as fr:
    model = pickle.load(fr)
    
def calculate_distance( pickup_longitude,pickup_latitude, dropoff_longitude,dropoff_latitude):
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude = map(np.radians, [pickup_longitude,pickup_latitude, dropoff_longitude,dropoff_latitude])

    # Find the differences
    longitude_diff = dropoff_longitude - pickup_longitude
    latitude_diff = dropoff_latitude - pickup_latitude

    # Apply the formula 
    a = np.sin(latitude_diff/2.0)**2 + np.cos(pickup_latitude) * np.cos(dropoff_latitude) * np.sin(longitude_diff/2.0)**2
    # Calculate the angle (in radians)
    angle = 2 * np.arcsin(np.sqrt(a))
    # Convert to kilometers
    distance_km =  6371.0072  * angle
    
    return distance_km
    
def main(): 
    
    pickup_longitude = st.slider("Input your pickup longitude",-74.0,-72.0)   
    pickup_latitude = st.slider("Input your pickup latitude",39.5,42.0)      
        
    dropoff_longitude = st.slider("Input your dropoff longitude",-75.0,-72.0)   
    dropoff_latitude = st.slider("Input your dropoff latitude",39.5,42.0)
    passenger_count = st.slider("Input your passenger count",1,6)  
    date1=st.date_input('time')
    hour = st.slider("Input hour: ",0,24)
    date1 = pd.to_datetime(date1,infer_datetime_format=True)
    weekday = date1.weekday()
    #month = date1.month()
    #year = date1.year()
    distance = calculate_distance(pickup_longitude,pickup_latitude, dropoff_longitude,dropoff_latitude)

    inputs = [[passenger_count,weekday, hour,distance]]

    if st.button('Predict'):
        result = model.predict(inputs)
        updated_res = result.flatten().astype(float)
        st.success("The predicted cost will be {}".format(updated_res))
   
if (__name__ == '__main__' ):
    main()  
