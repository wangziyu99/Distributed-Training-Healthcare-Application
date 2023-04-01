# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost as xgb

# Load the Random Forest CLassifier model
filename = 'regressor.json'
reg = xgb.XGBRegressor(tree_method="approx")
reg.load_model('regressor.json')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        num_lab_procedures = int(request.form['num_lab_procedures'])
        diag_1 = int(request.form['diag_1'])
        diag_2 = int(request.form['diag_2'])
        diag_3 = int(request.form['diag_3'])
        num_medications = int(request.form['num_medications'])
        age = int(request.form['age'])
        discharge_disposition_id = int(request.form['discharge_disposition_id'])
        medical_specialty = int(request.form['medical_specialty'])
        time_in_hospital = int(request.form['time_in_hospital'])
        num_procedures = int(request.form['num_procedures'])

        data = np.array([[num_lab_procedures, diag_1, diag_2, diag_3, num_medications, age,
                          discharge_disposition_id, medical_specialty, time_in_hospital, num_procedures]])

        my_prediction = (reg.predict(data) - 0.5 > 0).astype(int)

        print(my_prediction)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
