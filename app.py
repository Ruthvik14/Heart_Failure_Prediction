from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

model = joblib.load("heart_failure_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl") 
accuracy = joblib.load("accuracy.pkl") 

app = Flask(__name__)

EXTENSIONS = {'csv', 'xlsx', 'json', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS

@app.route('/')
def home():
    return "Heart Failure Prediction is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame([data['features']], columns=feature_names)
        if 'HeartDisease' in features.columns:
            features = features.drop(columns=['HeartDisease'])
        features_encoded = pd.get_dummies(features)
        features_encoded = features_encoded.reindex(columns=feature_names, fill_value=0)
        print(features_encoded)
        # features_encoded = features_encoded.iloc[:, :-1]

        features_scaled = scaler.transform(features_encoded)
        prediction = model.predict(features_scaled)
        return jsonify({
            'rf_accuracy': accuracy
            })
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']

        if file.filename== '':
            return jsonify({'error': 'No Selected File'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV, Excel, JSON, and TXT are allowed'}), 400
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.txt'):
            df = pd.read_csv(file, delimiter='\t')

       

        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
        # df_encoded = df_encoded.iloc[:, 1:]
        if 'HeartDisease' in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=['HeartDisease'])

        print(df_encoded.columns)


        print("Expected Features:", scaler.mean_.shape[0])
        print("Uploaded Data Shape:", df_encoded.shape[1])  

        if df_encoded.shape[1] != len(scaler.mean_):
            return jsonify({'error': f'Invalid number of features. Expected {len(scaler.mean_)} columns.'}), 400
        
        
        features_scaled = scaler.transform(df_encoded)
        predictions = model.predict(features_scaled)

        return jsonify({'rf_accuracy': accuracy})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ =='__main__':
    app.run(debug=True)