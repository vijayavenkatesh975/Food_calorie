import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Step 1: Initialize Flask app
app = Flask(__name__)
CORS(app)
# Step 2: Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="calorie_prediction_model.tflite")
interpreter.allocate_tensors()

# Get input and output details for the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper function to make predictions using the TFLite model
def predict_calories(protein, fat, carbs):
    # Prepare the input data as a numpy array
    input_data = np.array([[protein, fat, carbs]], dtype=np.float32)

    # Set the input tensor with the prepared data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the predicted calories from the output tensor
    predicted_calories = interpreter.get_tensor(output_details[0]['index'])

    return float(predicted_calories[0])

@app.route('/')
def home():
    return 'Home'
# Step 3: Define the route for calorie prediction (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract input features (Protein, Fat, Carbs) from the JSON request
    protein = float(data['Protein'])
    fat = float(data['Fat'])
    carbs = float(data['Carbs'])

    # Use the TFLite model to make a prediction
    predicted_calories = predict_calories(protein, fat, carbs)

    # Return the prediction as a JSON response
    response = {
        'Predicted Calories': predicted_calories
    }

    return jsonify(response)

# Step 4: Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
