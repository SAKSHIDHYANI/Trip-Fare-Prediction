#Import main library
import numpy as np
import os
import pandas as pd
import features.build_features

#Import Flask modules
from flask import Flask, request, render_template

#Import pickle to save our regression model
import pickle 

#Initialize Flask and set the template folder to "template"
app = Flask(__name__, template_folder = 'templates')
app.config['DEBUG'] = True
#Open our model 
d = os.path.dirname(os.getcwd())
path = d+"\\src\\models\\dtr_model_pkl"
model = pickle.load(open(path,'rb'))

#create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')

#Set a post method to yield predictions on page
@app.route('/', methods = ['POST'])
def predict():
    
    #obtain all form values and place them in an array, convert into integers
    features1 = [x for x in request.form.values()]
    #Combine them all into a final numpy array
    values = list(features1)
    for i in range(0,4):
        values[i] = float(values[i])
    date1=values[6]
    date1 = pd.to_datetime(date1,infer_datetime_format=True)
    weekday = date1.weekday()
    obj = features.build_features.Features()
    distance = obj.calculate_distance(values[0],values[1],values[2],values[3])
    direction = obj.calculate_direction(values[0],values[1],values[2],values[3])
    final_features =[]
    for i in range(0,4):
        final_features.append(values[i])
    final_features.append(float(values[4]))
    final_features.append(distance)
    final_features.append(direction)   
    final_features.append(int(values[5]))
    final_features.append(weekday)
      
    values = np.array(final_features)
    values = [values]
    
    #predict the price given the values inputted by user
    prediction = model.predict(values)
    
    #Round the output to 2 decimal places
    output = round(prediction[0], 4)
    
    #If the output is negative, the values entered are unreasonable to the context of the application
    #If the output is greater than 0, return prediction
    if output < 0:
        return render_template('index.html', prediction_text = "Predicted fare is negative, values entered not reasonable")
    elif output >= 0:
        return render_template('index.html', prediction_text = 'Predicted fare amount of the trip is: ${}'.format(output))   

#Run app
if __name__ == "__main__":
    app.run()