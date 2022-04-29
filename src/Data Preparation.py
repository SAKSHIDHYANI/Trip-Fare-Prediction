class DataPreparation:
    
    def __init__(self):
        pass
    
    def set_location_range(self,df):
        '''
           Confine longitude and latitude values to particular range
           As per the training data availability
           Latitude and longitude values can be modified in this function as per the training data availability
           
        '''
        df = df.loc[df['pickup_latitude'].between(39.5, 42)]
        df= df.loc[df['pickup_longitude'].between(-74, -72)]
        df = df.loc[df['dropoff_latitude'].between(39.5, 42)]
        df = df.loc[df['dropoff_longitude'].between(-75, -72)]
        return df
    
    def invalid_location(self,df,invalid_dropoff_longitude_max,invalid_dropoff_latitude_max,invalid_dropoff_latitude_min):
      '''
      
         To remove invalid locations as per training data.
         Outlier removal function can be changed according to the training data
         
      '''  
        
      df = df[(df['dropoff_longitude'] > invalid_dropoff_longitude_max) & (df['dropoff_latitude'] > invalid_dropoff_latitude_max) & (df['dropoff_latitude'] < invalid_dropoff_latitude_min)  ]
      return df
  
    def outlier_points(self):
        '''
           To remove outlier points using invalid location iteratively
        
        '''
        obj = DataPreparation()
        tripdata1 = obj.invalid_location(train,-74,40,40.5)
        train=train[~train.isin(tripdata1)].dropna(how = 'all')
        tripdata1 = obj.invalid_location(train,-73.75,40.5,40.8)
        train=train[~train.isin(tripdata1)].dropna(how = 'all')
        tripdata1 = obj.invalid_location(train,-73.61,40.5,41)
        train=train[~train.isin(tripdata1)].dropna(how = 'all')
  
    def invalid_distance(self,df):
        '''
           To remove invalid distance values
        '''
        remove_data = df[df['distance'] == 0 ]
        df = df[~df.isin(remove_data)].dropna(how = 'all')
        remove_data1 = df[((df['distance'] >= 0) & (df['distance'] <= 1) & (df['fare_amount'] > 100) )]
        df = df[~df.isin(remove_data1)].dropna(how = 'all')
        return df

        
    def invalid_fareamount(self,df):
        '''
        To remove trips having distance between 0 to 1 miles but having fare amount more than 100.
        
        '''
        remove_data = df[((df['distance'] >= 0) & (df['distance'] <= 1) & (df['fare_amount'] > 100) )]
        df = df[~df.isin(remove_data)].dropna(how = 'all')
        return df
    
    
    def invalid_passenger_count(self,df):
        
        '''
        To remove invalid passenger count
        
        '''
        df = df[df['passenger_count'] != 0]
        df = df[df['passenger_count'] <= 6]
        return df
  
        
  
    
    
       
      
   