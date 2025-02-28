from flask import Flask, request, jsonify, render_template
import joblib  # Use joblib to load the model
import pandas as pd

# Load the model
model = joblib.load('random.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json
        print("Received Data:", data)  # Debug: Print received data
        raw_features = data['features']  # Raw features as a list
        print("Raw Features:", raw_features)  # Debug: Print raw features

        # Convert raw features to a DataFrame
        feature_names = ['disrict', 'client_id', 'client_catg', 'region', 'creation_date', 
                         'tarif_type', 'counter_number', 'counter_code', 'reading_remarque', 
                         'consommation_level_1', 'counter_type']
        features_df = pd.DataFrame([raw_features], columns=feature_names)
        print("Features DataFrame:", features_df)  # Debug: Print DataFrame

        # Make a prediction
        prediction = model.predict(features_df)
        print("Prediction:", prediction)  # Debug: Print prediction

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print("Error:", str(e))  # Debug: Print error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)