import numpy as np

class Features:
    
    def __init__(self):
        pass
    
    def date_conversion(self,df):
        '''
        extracting new features using date column
        '''
        df['pickup_date'] = df['pickup_datetime'].dt.date
        df['pickup_weekday'] = df['pickup_datetime'].weekday()
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['pickup_year'] = df['pickup_datetime'].dt.year
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df.drop(['pickup_datetime'],axis = 1,inplace=True)
        df = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude', 'passenger_count', 'pickup_date','pickup_weekday', 'pickup_month', 'pickup_year', 'pickup_hour', 'fare_amount']]
        return df
    
    def calculate_distance(self,pickup_longitude,pickup_latitude, dropoff_longitude,dropoff_latitude):
        '''
         To calculate distance using latitude and longitude features
        '''
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
    
    def calculate_direction(self,lon1, lat1, lon2, lat2):
        ''' 
        function to calculate direction using latitude and longitude feature
        '''
              
        lon1=lon1
        lat1=lat1
        lon2=lon2
        lat2=lat2
        diff_lon = np.deg2rad(lon2-lon1)
        x = np.sin(diff_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_lon))
        initial_bearing = np.arctan2(x, y)

        # Now we have the initial bearing but math.atan2 return values
        # from -180° to + 180°.
        direction = np.degrees (initial_bearing)
        # Now we have the initial bearing but math.atan2 return values
        # from -180° to + 180° which is not what we want for a compass bearing
        # The solution is to normalize the initial bearing as shown below
        initial_bearing = np.degrees (initial_bearing)
        direction = (initial_bearing + 360) % 360
        return direction
            