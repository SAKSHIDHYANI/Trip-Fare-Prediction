
#importing libraries

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error 
from numpy import math
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle


# loading data
d = os.path.dirname(os.getcwd())
path = d+"\\data\\processed\\processed_data.csv"
tripdata = pd.read_csv(path)

# selecting the columns/features which will be used

train = tripdata[['passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude','pickup_weekday', 'pickup_hour','pickup_month','pickup_year','distance','direction','fare_amount']]

d = os.path.dirname(os.getcwd())
path = d+"\\data\\processed\\test_data.csv"
testdata = pd.read_csv(path)

# selecting columns 

test= testdata[['passenger_count','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'pickup_weekday', 'pickup_hour','distance', 'direction','fare_amount']] 

# defining independent variables which will be used for training

independent_variables = [ 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude','passenger_count','distance','direction','pickup_hour','pickup_weekday']

# function to print rmse as metric value for different models

RMSE_list = []

def print_metrics(actual, predicted,independent_variables):
  '''
     function to print rmse as metric value for different models

  '''
  print('RMSE is {}'.format(math.sqrt(mean_squared_error(actual, predicted))))
  RMSE_list.append(math.sqrt(mean_squared_error(actual, predicted)))

# calculating average trip fare amount as predicted value for baseline model

avg_fare=round(np.mean(train['fare_amount']),2)
baseline_pred=np.repeat(avg_fare,test['fare_amount'].shape[0])
baseline_rmse=np.sqrt(mean_squared_error(baseline_pred, test['fare_amount']))
print("Baseline RMSE of Validation data :",baseline_rmse)


# Decision Tree Regressor

dtregressor = DecisionTreeRegressor(random_state=0)
dtregressor.fit(train[independent_variables],train['fare_amount'])
y_test_predicted = dtregressor.predict(test[independent_variables])
y_train_pred = dtregressor.predict(train[independent_variables])

# RMSE vale for training and testing data

print("Evaluation metrics for training data")
print("--------------------------------------")
print_metrics(train['fare_amount'],y_train_pred,independent_variables)
print("--------------------------------------")
print("Evaluation metrics for test data")
print("--------------------------------------")
print_metrics(test['fare_amount'], y_test_predicted,independent_variables)

# hyper parameter tuning using grid search CV 

parameters={"splitter":["best","random"],
            "min_samples_leaf":[1,2,3,4,5,6,7],
           "min_samples_split":[2,3,4,5,6],
            "max_depth":[4,5,6]
           }

dtregressor = DecisionTreeRegressor(random_state=0)
tuning_model=GridSearchCV(dtregressor,param_grid=parameters,cv=3,verbose=3)
tuning_model.fit(train[independent_variables],train['fare_amount'])
y_test_predicted = tuning_model.predict(test[independent_variables])
y_train_pred = tuning_model.predict(train[independent_variables])

# parameters selected by model while performing hyper parameter tuning

print(tuning_model.best_params_)

# RMSE value for training and test data after performing hyper parameter tuning to the decision tree regressor model

print("Evaluation metrics for training data")
print("--------------------------------------")
print_metrics(train['fare_amount'],y_train_pred,independent_variables)
print("--------------------------------------")
print("Evaluation metrics for test data")
print("--------------------------------------")
print_metrics(test['fare_amount'], y_test_predicted,independent_variables)


#Gradient Boost Regressor

GB=GradientBoostingRegressor()
GB.fit(train[independent_variables],train['fare_amount'])

# RMSE values

y_test_predicted = GB.predict(test[independent_variables])
y_train_pred = GB.predict(train[independent_variables])
print("Evaluation metrics for training data")
print("--------------------------------------")
print_metrics(train['fare_amount'],y_train_pred,independent_variables)
print("--------------------------------------")
print("Evaluation metrics for test data")
print("--------------------------------------")
print_metrics(test['fare_amount'], y_test_predicted,independent_variables)

print(GB.get_params())

# Hyper parameter tuning the Gradient Boosting regressor model

GBR=GradientBoostingRegressor()
search_grid={"alpha":[0.1,0.01],'learning_rate': [0.1, 0.01],'n_estimators':[70,80,90],'max_depth':[2]}
search=GridSearchCV(estimator=GBR,param_grid=search_grid,n_jobs=-1,cv=5)
search.fit(train[independent_variables],train['fare_amount'])

# checking best parameters
print(search.best_params_)

# Evaluation metric for tuned gradient boost regressor model

y_test_predicted = search.predict(test[independent_variables])
y_train_pred = search.predict(train[independent_variables])
print("Evaluation metrics for training data")
print("--------------------------------------")
print_metrics(train['fare_amount'],y_train_pred,independent_variables)
print("--------------------------------------")
print("Evaluation metrics for test data")
print("--------------------------------------")
print_metrics(test['fare_amount'], y_test_predicted,independent_variables)

# create an iterator object with write permission - model.pkl
# saving gradient boost regressor model
with open('gbr_model_pkl', 'wb') as files:
    pickle.dump(search, files)
# saving decision tree regressor model

with open('dtr_model_pkl', 'wb') as files:
    pickle.dump(tuning_model, files)