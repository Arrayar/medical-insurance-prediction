from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import json
import os
from pathlib import Path

app = Flask(__name__,
            static_folder='../Client/static',
            template_folder='../Client')

# Get absolute paths to model files
base_dir = Path(__file__).parent.parent
model_path = base_dir / 'Model' / 'insurance_model.pkl'
columns_path = base_dir / 'Model' / 'columns.json'

# Load model and columns
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(columns_path, 'r') as f:
    data_columns = json.load(f)['data_columns']

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Preprocess inputs
        sex = 0 if sex == 'male' else 1
        smoker = 0 if smoker == 'yes' else 1
        region_mapping = {'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}
        region = region_mapping[region]
        
        # Feature engineering
        bmi_age = bmi * age
        children_smoker = children * smoker
        
        # Create input array
        input_array = np.array([age, sex, bmi, children, smoker, region, bmi_age, children_smoker])
        input_array = input_array.reshape(1, -1)
        
        # Predict
        prediction = model.predict(input_array)[0]
        
        return render_template('app.html',
                            prediction_text=f"Predicted Insurance Cost: ${round(prediction, 2)}")
    
    except Exception as e:
        return render_template('app.html',
                            prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)