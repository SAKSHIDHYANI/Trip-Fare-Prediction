#!/usr/bin/env python
# coding: utf-8

# #                                     **Fare Amount Prediction**
# 
# The dataset “trips.csv” contains the following fields:
# 
# **key** - a unique identifier for each trip
# 
# 
# **fare_amount** - the cost of each trip in usd
# 
# **pickup_datetime** - date and time when the meter was engaged
# 
# **passenger_count** - the number of passengers in the vehicle (driver entered value)
# 
# **pickup_longitude** - the longitude where the meter was engaged
# 
# **pickup_latitude** - the latitude where the meter was engaged
# 
# **dropoff_longitude**- the longitude where the meter was disengaged
# 
# **dropoff_latitude** - the latitude where the meter was disengaged
# 
# # **– You need to analyse the data and create an efficient model that will estimate the fare prices accurately.**
# 

# ## **Importing the required Libraries**

# In[1]:


# importing libraries

import pandas as pd
from bokeh.io import export_png
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import ssl
from math import cos, asin, sqrt, pi
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## **Reading Data and Examining Features**

# In[2]:


# loading data
tripdata = pd.read_csv(r"C:\Users\saksh\venv\Trip Fare Amount Prediction\data\raw\trip.csv")
tripdata.head()


# In[3]:


# checking columns in the dataset

tripdata.columns


# In[4]:


# adjusting the columns sequence

tripdata = tripdata[['index', 'key',  'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count','fare_amount']]


# In[5]:


# checking the top 5 rows of the data

tripdata.head()


# In[6]:


# shape of data (train and test combined)

tripdata.shape


# In[7]:


# information for features of the dataset

tripdata.info()


# ## **Describe Data**

# In[8]:


# checking statistical data for the features of the dataset

tripdata[['pickup_longitude','pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','passenger_count','fare_amount']].describe()


# ## **Hypothesis Generation**
# 
# We are making some assumptions based on what we expect from data. We will perform hypothesis validation in further steps.
# 
# 1. Fare Amount increased over the years.
# 2. Number of trips increases during peak hours
# 3. Fare amount is more for unusual hours
# 4. Fare amount will be more on weekends.
# 5. Fare Amount increases as distance increases.

# ## **Data Exploration , Data Cleaning and Feature Engineering**

# In[9]:


# checking for null values

tripdata.isna().sum()


# **Since there is only 1 null value for two of the features, we will remove those rows containg null values**

# In[10]:


# dropping rows with null values for features

tripdata.dropna(axis=0, how='any',  subset=['dropoff_longitude', 'dropoff_latitude'], inplace=True)


# In[11]:


# validating if null values are removed

tripdata.isna().sum()


# In[12]:


# checking shape of data after removing null values

tripdata.shape


# In[13]:


# converting pickup date time format to proper date format

tripdata['pickup_datetime'] = pd.to_datetime(tripdata['pickup_datetime'],format = '%Y-%m-%d %H:%M:%S UTC')


# In[14]:


# validating pickup datetime format changes

tripdata.head()


# **Here, we are creating different columns for month, year and weekdays to validate our hypothesis. Also, we have dropped index and key columns as they are not required.**

# In[15]:


# creating different columns for pickup date time feature : weekdays, month,year, hour . These features will help in further data exploration as well.


tripdata.drop(['index','key'],axis = 1,inplace=True)
tripdata['pickup_date'] = tripdata['pickup_datetime'].dt.date
tripdata['pickup_weekday'] = tripdata['pickup_datetime'].dt.day_name()
tripdata['pickup_month'] = tripdata['pickup_datetime'].dt.month
tripdata['pickup_year'] = tripdata['pickup_datetime'].dt.year
tripdata['pickup_hour'] = tripdata['pickup_datetime'].dt.hour
tripdata.drop(['pickup_datetime'],axis = 1,inplace=True)
tripdata = tripdata[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude', 'passenger_count', 'pickup_date','pickup_weekday', 'pickup_month', 'pickup_year', 'pickup_hour', 'fare_amount']]
tripdata.head()


# In[16]:


# function to plot total count for various categories of features

def plot(col_name):
  tripdata.groupby([col_name])[col_name].count().plot(kind="bar",color='g')
  plt.ylabel("Count Of Trips")
  plt.title("Count of trips Vs. {}".format(col_name),fontweight="bold")
  plt.figure(figsize=(15,15)) 


# ## **Count of Trips Vs Pickup Month**

# In[17]:



plot('pickup_month')


# **For Months March,May,January,June, people tend to travel more as compared to other months.**

# ## **Count of Trips Vs pickup year**

# In[18]:


plot('pickup_year')


# **The trips are slightly more for years 2011 and 2012**

# ## **Count of Trips Vs pickup hours**

# In[19]:


plot('pickup_hour')


# **It can be observed that the number of trips increases for the peak hours like 5 pm, 6 pm. This also validates our one of the hypothesis.** 
# 

# ## **Trip count Vs pickup weekday**

# In[20]:


plot('pickup_weekday')


# **For Fridays, Saturdays trips are more**

# ## **Count of Trips Vs Passenger Count**

# In[21]:


plot('passenger_count')


# **Passenger count is one for most of the rides. 208 paseenger count is a noisy data point as it is not possible to have 208 as passenger count. We will remove this data point once we split the data into training and test data.**

# ## **Target Variable : Fare Amount Distribution** 

# In[22]:


# fare amount density

plt.figure(figsize=(10,6))
sns.histplot(tripdata['fare_amount'], color="green", kde=True, stat="density", linewidth=0)


# In[23]:


tripdata['fare_amount'].hist(bins=70, figsize=(8,5),range=(0, 100))
plt.xlabel('fare')


# In[24]:


print(f"There are {len(tripdata[tripdata['fare_amount'] < 0])} negative fares.")
print(f"There are {len(tripdata[tripdata['fare_amount'] == 0])} $0 fares.")
print(f"There are {len(tripdata[tripdata['fare_amount'] > 100])} fares greater than $100.")


# In[25]:


# specifically for checking maximum and minimum values for fare amount

tripdata[['fare_amount']].describe()


# **Performing train test split. We will be first processing training data. During model validation, we will handle test data data transformation if required.**

# In[26]:


# Train Test split

train, test = train_test_split(tripdata, test_size = 0.2, random_state = 0)


# ## **Feature binning**

# In[27]:


# creating feature binning for categorizing target variable fare amount 

bins = [0,5,10,15,20,25,30,35,40,45,50,np.inf]
train['farebins'] = pd.cut(train['fare_amount'], bins)


# In[28]:


# checking count for different farebins

train['farebins'].value_counts()


# ## **Fare Bins Count**

# In[29]:


# Fare bins count 

train['farebins'].value_counts().sort_index().plot.bar(color = 'b', edgecolor = 'k',figsize=(8,5));
plt.title('Categorized fares');


# **The fare amount mostly lies in between 5-10 range followed by 10-15 and 0-5**

# ## **Outlier Detection and Removal**

# In[30]:


# Empirical Cumulative Distribution Function Plot for target variable 

x = np.sort(train['fare_amount'])
n = len(train['fare_amount'])
y = np.arange(1, n + 1, 1) / n
plt.figure(figsize = (9, 6))
plt.plot(x, y, '.')
plt.ylabel('Percentile'); plt.title('ECDF of Fare Amount'); plt.xlabel('Fare Amount ($)');


# **ECDF helps in identifying outliers as well, we can see fare amount outliers as well near 500 and 300**

# In[31]:


# checking trips having more than 200 fare amount

train[(train['fare_amount'] > 200)]


# **We won't be removing fare amount greater than 200 right now, as we will calculate distance and check if these fare amounts are justified or not.**

# **Before proceeding with the outlier removal for any other features, we will try to remove outliers for pick up and drop off location. There might be possibility that outliers for other features are removed while dealing with location wise outliers.**

# In[32]:


# checking trips having latitude and longitude values equal to 0

train[(train.pickup_latitude==0) | (train.pickup_longitude)==0 | (train.dropoff_latitude==0)|(train.dropoff_longitude==0)].shape


# In[33]:


# checking statistical details specifically for longitude and latitude values

train.describe()


# ## **Plot for dropoff Locations and pickup Locations**

# In[34]:


# scatter plot for pickup and drop off locations

warnings.filterwarnings('ignore')
fig, axes = plt.subplots(1, 2, figsize = (12, 5), sharex=True, sharey=True)
axes = axes.flatten()
# Plot Longitude (x) and Latitude (y)
g = sns.scatterplot('pickup_longitude', 'pickup_latitude', 
            data = train, ax = axes[0],color='green' );
k =sns.scatterplot('dropoff_longitude', 'dropoff_latitude',  
            data = train, ax = axes[1]);
axes[0].set_title('Pickup Locations')
axes[1].set_title('Dropoff Locations');
city_long_border = (-74.70, -73.30)
city_lat_border = (40.15, 41.40)
g.set(ylim=city_lat_border)
g.set(xlim = city_long_border )


# **This plot does not give much clear idea about pickup and drop off locations. For proper visualization, we will plot locations on the actual map.**

# ## **Pickup and Dropoff Locations on map**

# In[35]:


# plotting locations for pickup and drop off on the actual map.
import ssl
axis = (-74.20, -73.70, 40.50, 40.90)
import certifi
import imageio
#ssl._create_default_https_context = ssl._create_unverified_context
nyc_map = imageio.imread(r'C:\Users\saksh\venv\Trip Fare Amount Prediction\notebooks\nyc_map.png')

def plot_on_map(df,nyc_map, color = False):
    fig, axs = plt.subplots(1, 2, figsize=(22, 18))
    axs[0].scatter(df.pickup_longitude, df.pickup_latitude, zorder=1, alpha=0.05, c='g', s=0.05)
    axs[0].set_xlim((axis[0], axis[1]))
    axs[0].set_ylim((axis[2], axis[3]))
    axs[0].set_title('Pickup locations')
    axs[0].imshow(nyc_map, zorder=0, extent=axis)

    axs[1].scatter(df.dropoff_longitude, df.dropoff_latitude, zorder=1, alpha=0.05, c='b', s=0.05)
    axs[1].set_xlim((axis[0], axis[1]))
    axs[1].set_ylim((axis[2], axis[3]))
    axs[1].set_title('Dropoff locations')
    axs[1].imshow(nyc_map, zorder=0, extent=axis)
    

plot_on_map(train,nyc_map)


# **As per statistical data, the minimum and maximum values are very high  or low according to the most of the latitude and longitude range. We have selected range for latitude and longitude values according to the statistical data and the map plotted above**

# In[36]:


# confining training data to have latitude and longitude values between specified range.

train = train.loc[train['pickup_latitude'].between(39.5, 42)]
train= train.loc[train['pickup_longitude'].between(-74, -72)]
train = train.loc[train['dropoff_latitude'].between(39.5, 42)]
train = train.loc[train['dropoff_longitude'].between(-75, -72)]
print(f'New number of observations: {train.shape[0]}')


# In[37]:


# function to plot farebins vs locations on the map

def plot_map(col_x="pickup_longitude" , col_y= "pickup_latitude",col_name = 'farebins'):
  sns.set(rc={'figure.figsize':(16,16)})
  axis = (-75.20, -73.40, 38.50, 42.89)
  sns.scatterplot(data=train, x=col_x, y=col_y,hue= col_name,palette="flare")
  plt.imshow(nyc_map, zorder=0, extent=axis, aspect='auto')
  


# ## **Fare Bins Vs pickup location**

# In[38]:


plot_map()


# ## **Fare Bins Vs drop off locations**

# In[39]:


plot_map("dropoff_longitude","dropoff_latitude")


# **As expected in both the maps, most of the data points are having fare amount within range 0-5 or 5-10**

# **Another observations that we have is, many data points are lying in invalid locations like many data points are lying in water. That seems to be noisy data. we will remove the data points that are lying in water**

# In[40]:


# function to remove data points which seems to be noisy as per the location in the map

def invalid_location(original_data,m,n,p):
  df = original_data[(original_data['dropoff_longitude'] > m) & (original_data['dropoff_latitude'] > n) & (original_data['dropoff_latitude'] < p)  ]
  return df


# In[41]:


# Removing data points lying in water using the latitude and longitude values that are visible in map for those outlier data points

tripdata1 = invalid_location(train,-74,40,40.5)
train=train[~train.isin(tripdata1)].dropna(how = 'all')
tripdata1 = invalid_location(train,-73.75,40.5,40.8)
train=train[~train.isin(tripdata1)].dropna(how = 'all')
tripdata1 = invalid_location(train,-73.61,40.5,41)
train=train[~train.isin(tripdata1)].dropna(how = 'all')


# In[42]:


# shape of data after removing points lying in water

train.shape


# ## **Map Plot for dropoff locations after removing invalid locations**

# In[43]:


# validating if points lying in water are removed for dropoff locations

plot_map("dropoff_longitude","dropoff_latitude")


# ## **Map Plot for pickup locations after removing invalid locations**

# In[44]:


# validating if points lying in water are removed for pickup locations

plot_map()


# **Now since location wise, we have removed some outliers. We will now focus on calculating distance using latitude and longitude locations.**

# In[45]:


# function to calculate distance in kilometers using latitude and longitude locations.Distance between points using Haversine distance

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


# **Adding new column distance**

# In[46]:


# applying function to all the rows of the dataframe to add new column

train['distance'] =  train.apply(lambda row:calculate_distance(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']),axis=1)


# **After calculating distance, we can remove trips having 0 distance. Also, as discussed above , we cannot remove the large fare amounts directly. so for reasoning, we will be removing trips having fare amounts which are more than 100 for trip distance between 0 to 1 miles, as this can not be possible to have such high fare for such shorter distance. Even if we compare with other records, most of the trips are having fare amount around 5 - 10 or 0-5 range even for distances more than 1 mile.**

# In[47]:


# removing trips having distance value equal to zero

remove_data = train[train['distance'] == 0 ]
train = train[~train.isin(remove_data)].dropna(how = 'all')

# removing trips having distance between 0 to 1 miles but having fare amount more than 100 for that range.

remove_data1 = train[((train['distance'] >= 0) & (train['distance'] <= 1) & (train['fare_amount'] > 100) )]
train = train[~train.isin(remove_data1)].dropna(how = 'all')
train.shape


# ## **Distance Vs Binned Fare Amount**

# In[48]:


# plotting binned fare amount vs Distance graph 

sns.barplot(data=train, x="farebins", y="distance")
sns.set(rc = {'figure.figsize':(10,8)})
plt.xticks(rotation=45)


# **We have high fare amount for trips having larger distances.This also validates our one of the assumptions.**

# **Checking if more outliers can be detected**

# ## **Box plots for remaining outlier detection**

# In[49]:


# box plot for detection of remaining outliers if left

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize=(25,4))
sns.boxplot(x=train['pickup_longitude'],ax=ax1,color = 'orange')
sns.boxplot(x=train['pickup_latitude'],ax=ax2,color = 'orange')
sns.boxplot(x=train['dropoff_longitude'],ax=ax3,color = 'orange')
sns.boxplot(x=train['dropoff_latitude'],ax=ax4,color = 'orange')


# In[50]:


# box plot for detection of remaining outliers if left

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(25,4))
sns.boxplot(x=train['passenger_count'],ax=ax1,color = 'orange')
sns.boxplot(x=train['distance'],ax=ax2,color = 'orange')


# In[51]:


train['passenger_count'].value_counts()


# In[52]:


print("Number of trips having passenger count as 0: ",train[train['passenger_count'] == 0].shape[0])


# In[53]:


# Removing the records having passenger count as 0

train = train[train['passenger_count'] != 0]


# In[54]:


train.shape


# In[55]:


print("trips having distance more than 130 miles is",train[train['distance']>132].shape[0])
print("Maximum distance value in the training data: ",max(train['distance']))


# **We removed trips having passenger count as 0, and for distance, we have maximum value of fare amount equal to about 130. we have already removed trips with fare amount more than 150 while dealing with other outliers.**

# In[56]:


train.columns


# In[57]:


# converting weekdays to numerical features

train['pickup_weekday'] = pd.to_datetime(train['pickup_date']).apply(lambda x: x.weekday())


# In[58]:


train['fare_amount_per_km'] = train['fare_amount']/train['distance']


# In[59]:


train.head()


# In[60]:


def plot_mean_fareamt(col):
    df = pd.DataFrame(train.groupby([col])['fare_amount_per_km'].mean())
    df.reset_index(inplace=True)
    plt.rcParams['figure.figsize'] = [5, 5]
    sns.barplot(x=col, y='fare_amount_per_km', data=df)
    plt.ylabel("Mean fare_amount_per_km for all trips for particular {}".format(col))


# ## **Mean Fare Amount per Km Vs Year**

# In[61]:


#display graph for Mean fare amount per km for different years

plot_mean_fareamt("pickup_year")


# **Plot for Mean fare amount per km calculated for all trips for specific years shows that for year 2010 and 2009 it's maximum. it decreased a lot in between specifically for 2011 year and 2013 year. then it showed sudden rise in year 2014.**

# ## **Mean Fare Amount per Km Vs Month**

# In[62]:


#display graph for Mean fare amount per km for different months

plot_mean_fareamt("pickup_month")


# **Plot for Mean fare amount per km calculated for all trips for specific months shows that for month of April,May and August, it's maximum**

# ## **Mean Fare Amount per Km Vs Weekday**

# In[63]:


#display graph for Mean fare amount per km for different weekdays

plot_mean_fareamt("pickup_weekday")


# **Plot for Mean fare amount per km calculated for all trips for specific weekday shows that for wednesday and Sunday it's maximum**

# ## **Mean Fare Amount per Km Vs Hours**

# In[64]:


#display graph for Mean fare amount per km for different hours
bins = [0,5,11,16,20,23]
train['hourbins'] = pd.cut(train['fare_amount'], bins)
plot_mean_fareamt("hourbins")


# **Plot for Mean fare amount per km calculated for all trips for specific hours shows that between 1 am to 5 am the mean fare amount is maximum even more than the peak hours, it can be due to unusual hours that taxis are charging high fare amount.**

# In[65]:


from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure, ColumnDataSource
from bokeh.tile_providers import get_provider, Vendors
from bokeh.palettes import PRGn, RdYlGn
from bokeh.transform import linear_cmap,factor_cmap
from bokeh.layouts import row, column
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter
import numpy as np
import pandas as pd


# In[66]:


train.columns


# In[67]:


def x_coord(x, y):
    
    lat = x
    lon = y
    
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
        lat * (np.pi/180.0)/2.0)) * scale
    return (x, y)
# Define coord as tuple (lat,long)
train['coordinates'] = list(zip(train['pickup_latitude'], train['pickup_longitude']))
# Obtain list of mercator coordinates
mercators = [x_coord(x, y) for x, y in train['coordinates'] ]


# In[68]:


# Create mercator column in our df
train['mercator'] = mercators
# Split that column out into two separate columns - mercator_x and mercator_y
train[['mercator_x', 'mercator_y']] = train['mercator'].apply(pd.Series)


# In[69]:


#saving old data in train
keep_train = train


# In[70]:


train = train.drop(['farebins','hourbins'],axis=1)


# In[71]:


get_ipython().system(' pip install bokeh')


# In[72]:


import bokeh
def bokeh_plot_hour(train,from_hour,to_hour,year):
    train =  train[train['pickup_year'] == year]
    train = train[train['pickup_hour'] >= from_hour ]
    train = train[train['pickup_hour'] <= to_hour ]

    # Select tile set to use
    chosentile = get_provider(Vendors.STAMEN_TONER)
    # Choose palette
    palette = PRGn[11]
    # Tell Bokeh to use train as the source of the data
    source = ColumnDataSource(data=train)
    # Define color mapper - which column will define the colour of the data points
    color_mapper = linear_cmap(field_name = 'fare_amount_per_km', palette = palette, low = train['fare_amount_per_km'].min(), high = train['fare_amount_per_km'].max())
    # Set tooltips - these appear when we hover over a data point in our map, very nifty and very useful
    tooltips = [("Fare Amount per km","@fare_amount_per_km"),("Pickup Month","@pickup_month"),("Pickup weekday","@pickup_weekday"),("Pickup Hour","@pickup_hour")]
    # Create figure
    p = figure(title = 'Fare Amount per km by region for hours '+str(from_hour)+"-"+ str(to_hour), x_axis_type="mercator", y_axis_type="mercator", x_axis_label = 'pickup_longitude', y_axis_label = 'pickup_latitude', tooltips = tooltips)
    # Add map tile
    p.add_tile(chosentile)
    # Add points using mercator coordinates
    p.circle(x = 'mercator_x', y = 'mercator_y', color = color_mapper, source=source, size=10, fill_alpha = 0.7)
    #Defines color bar
    color_bar = ColorBar(color_mapper=color_mapper['transform'], 
                        formatter = NumeralTickFormatter(format='0.0[0000]'), 
                        label_standoff = 14, width=2, location=(0,0))
    # Set color_bar location
    p.add_layout(color_bar, 'right')
    # Display in notebook
    output_notebook()
    # Save as HTML
    name = 'fareamountperkm_'+str(from_hour)
    output_file(name, title='Fare Amount per km by region for hour range'+str(from_hour)+"-"+str(to_hour))
    # Show map
    show(p)


# In[73]:


bokeh_plot_hour(train,17,22,2015)


# In[74]:


bokeh_plot_hour(train,0,6,2015)
bokeh_plot_hour(train,17,22,2015)
bokeh_plot_hour(train,9,12,2015)


# **To visualize in map, we can use Bokeh to visualize  the trips for different hours based on pickup location. For first plot, we were able to see the pickup locations for evening hours (5 pm to 10 pm ). For second plot,we were able to see the pickup locations for unsual hours(12 am to 6 am). By hovering cursor on the locations, we can compare the fare amount for different hours and other details availble in tooltip.**

# In[75]:


train = keep_train
train['delta_lon'] = train.pickup_longitude - train.dropoff_longitude
train['delta_lat'] = train.pickup_latitude - train.dropoff_latitude

# direction of a trip, from 180 to -180 degrees. Horizontal axes = 0 degrees.
def calculate_direction(d_lon, d_lat):
    result = np.zeros(len(d_lon))
    l = np.sqrt(d_lon**2 + d_lat**2)
    result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
    idx = (d_lon<0) & (d_lat>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    idx = (d_lon<0) & (d_lat<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
    return result

train['direction'] = calculate_direction(train.delta_lon, train.delta_lat)


# In[76]:


train.head()


# ## **Direction VS fare Amount**

# In[77]:


train['fare_amount_per_km'].describe()


# In[78]:


train.shape


# In[79]:


train = train[(train['fare_amount_per_km']>0) & (train['fare_amount_per_km']<=80) ]
train.shape


# In[80]:


min(train['fare_amount_per_km'])


# In[81]:


sns.scatterplot(data=train, x="direction", y="fare_amount_per_km")


# In[82]:


sns.scatterplot(data=train, x="direction", y="fare_amount")


# **There seems to be some impact of direction on target variable fare amount**
