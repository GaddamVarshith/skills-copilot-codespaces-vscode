from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('your_model.pkl')  # Replace 'your_model.pkl' with your actual model file path

# Route to render index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        features = request.get_json()

        # Convert data to numpy array and reshape
        features_array = np.array(list(features.values())).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)

        # Prepare response as JSON
        result = {
            'prediction': prediction.tolist()  # Convert numpy array to Python list
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
