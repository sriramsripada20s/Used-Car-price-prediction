from flask import Flask, render_template, request , jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('carprice_rfr_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':

        year = int(request.form['year'])

        condition = request.form['condition']
        if(condition=='with mileage'):
            condition = 2
        elif(condition=='for parts'):
            condition = 1
        else:
            condition = 0


        mileage_km = float(request.form['mileage_km'])
        mileage1 =np.log1p(mileage_km)

        fuel_type = request.form['fuel_type']
        if(fuel_type=='electrocar'):
            fuel_type=0
        elif(fuel_type=='petrol'):
            fuel_type = 1
        else:
            fuel_type = 2

        volume_cm3 = float(request.form['volume_cm3'])
        volume1 =np.log1p(volume_cm3)

        color = request.form['color']
        if(color=='black'):
            color=12
        elif(color=='silver'):
            color = 11
        elif(color=='blue'):
            color=10
        elif(color=='gray'):
            color= 9
        elif(color=='white'):
            color= 8
        elif(color=='green'):
            color = 7
        elif(color=='other'):
            color=6
        elif(color=='red'):
            color = 5
        elif(color=='burgundy'):
            color = 4
        elif(color=='brown'):
            color = 3
        elif(color=='purple'):
            color=2
        elif(color=='yellow'):
            color = 1                   
        else:
            color = 0

        no_of_year = 2020 - year 


        transmission = request.form['transmission']
        if(transmission=='mechanics'):
            transmission=1
        else:
            transmission=0

        drive_unit= request.form['drive_unit']
        if(drive_unit=='front-wheel drive'):
            drive_unit=0
        elif(drive_unit=='all-wheel drive'):
            drive_unit=1
        elif(drive_unit=='rear drive'):
            drive_unit=2
        else:
            drive_unit = 3




        segment = request.form['segment']
        if(segment=='A'):
            segment = 1
        elif(segment=='B'):
            segment= 2
        elif(segment=='C'):
            segment = 3
        elif(segment=='D'):
            segment= 4
        elif(segment=='E'):
            segment = 5
        elif(segment=='F'):
            segment= 6
        elif(segment=='J'):
            segment = 7
        elif(segment=='M'):
            segment= 8
        else:
            segment = 9



        prediction=model.predict([[condition,mileage1,volume1,color,year,fuel_type,no_of_year,transmission,drive_unit,segment]])
        prediction_normal = np.expm1(prediction)

        output=round(prediction_normal[0],2)

        if output<0:
            return render_template('index.html',prediction_texts="Sorry Car price cannot be predicted")
        else:
            return render_template('index.html',prediction_text="Used car price is {} USD".format(output))
        
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)