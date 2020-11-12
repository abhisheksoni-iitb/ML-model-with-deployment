import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import datetime

def car_age(model_year):
    age_of_car = datetime.datetime.now().year - model_year
    return (10*(age_of_car-1)/(62))

def distance_normalize(distance):
    return (distance)*10/9999999.0

def volume_normalize(volume):
    return (volume-500.0)*10/(20000.0 - 500.0)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


dictionaries = [{'for parts': 0, 'with damage': 1, 'with mileage': 2},
                {'diesel': 0, 'petrol': 1},
                {'black': 0,
                'blue': 1,
                'brown': 2,
                'burgundy': 3,
                'gray': 4,
                'green': 5,
                'orange': 6,
                'other': 7,
                'purple': 8,
                'red': 9,
                'silver': 10,
                'white': 11,
                'yellow': 12},
                {'auto': 0, 'mechanics': 1},
                {'all-wheel drive': 0,
                'front-wheel drive': 1,
                'part-time four-wheel drive': 2,
                'rear drive': 3},
                {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'J': 6, 'M': 7, 'S': 8}]



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    
    
    dictionary = {
                    'year':[car_age(features[0])],
                    'condition':[features[1]],
                    'mileage(kilometers)':[distance_normalize(features[2])],
                    'fuel_type':[features[3]],
                    'volume(cm3)':[volume_normalize(features[4])],
                    'color':[features[5]],
                    'transmission':[features[6]],
                    'drive_unit':[features[7]],
                    'segment':[features[8]]
                }
    final_features = pd.DataFrame(data=dictionary)
    
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Estimated price should $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)