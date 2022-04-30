# importing libraries

import features.build_features
import data.data_preprocess
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# loading data

d = os.path.dirname(os.getcwd())
path = d+"\\data\\raw\\trip.csv"
tripdata = pd.read_csv(path)

# adjusting the columns sequence

tripdata = tripdata[['index', 'key',  'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count','fare_amount']]

# dropping rows with null values for features

tripdata.dropna(axis=0, how='any',  subset=['dropoff_longitude', 'dropoff_latitude'], inplace=True)

# converting pickup date time format to proper date format

tripdata['pickup_datetime'] = pd.to_datetime(tripdata['pickup_datetime'],format = '%Y-%m-%d %H:%M:%S UTC')

# creating different columns for pickup date time feature : weekdays, month,year, hour . These features will help in further data exploration as well.

# using module features.build features

obj  = features.build_features.Features()
tripdata = obj.date_conversion(tripdata)

# Train Test split

train, test = train_test_split(tripdata, test_size = 0.2, random_state = 0)

# confining training data to have latitude and longitude values between specified range.
# using module data.data_preprocess

obj_data = data.data_preprocess.PreProcess()
train = obj_data.set_location_range(train)


# Removing data points lying in water using the latitude and longitude values that are visible in map for those outlier data points

train = obj_data.outlier_points(train)

# applying function to all the rows of the dataframe to add new column for distance

train['distance'] =  train.apply(lambda row:obj.calculate_distance(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']),axis=1)

# removing trips having distance value equal to zero

train = obj_data.invalid_distance(train)

#removing invalid passenger count

train = obj_data.invalid_passenger_count(train)

# creating new column having direction value for trip

train['direction'] = obj.calculate_direction(train['pickup_latitude'], train['pickup_longitude'],train['dropoff_latitude'], train['dropoff_longitude'])

# creating same columns as train data in test data using functions

test['distance'] =  test.apply(lambda row:obj.calculate_distance(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']),axis=1)
test['direction'] = obj.calculate_direction(test['pickup_latitude'], test['pickup_longitude'],test['dropoff_latitude'], test['dropoff_longitude'])
test['pickup_weekday'] = pd.to_datetime(test['pickup_date']).apply(lambda x: x.weekday())


# saving data into new files after preprocessing and splitting

test.to_csv('test_data.csv')
train.to_csv('processed_data.csv')