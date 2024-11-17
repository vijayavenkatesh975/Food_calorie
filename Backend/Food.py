import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sys

# Step 1: Reconfigure stdout to use UTF-8 encoding to avoid UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Step 2: Load the Dataset
dataset_path = r"C:\Vijay\Program\nutrients_csvfile.csv"  # Update this to your dataset path
dataset = pd.read_csv(dataset_path)

# Step 3: Convert columns to numeric and handle errors
dataset['Protein'] = pd.to_numeric(dataset['Protein'], errors='coerce')
dataset['Fat'] = pd.to_numeric(dataset['Fat'], errors='coerce')
dataset['Carbs'] = pd.to_numeric(dataset['Carbs'], errors='coerce')
dataset['Calories'] = pd.to_numeric(dataset['Calories'], errors='coerce')

# Drop rows with missing values
dataset = dataset.dropna()

# Step 4: Prepare the data for training
X = dataset[['Protein', 'Fat', 'Carbs']]  # Features
y = dataset['Calories']  # Labels

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Model
model = Sequential()

# Input layer (3 inputs: Protein, Fat, Carbs)
model.add(Dense(64, input_shape=(3,), activation='relu'))

# Hidden layer
model.add(Dense(32, activation='relu'))

# Output layer (1 output: Calories)
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Step 6: Evaluate the Model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Step 7: Save the Model
model.save('calorie_prediction_model.h5')
print("Model saved as 'calorie_prediction_model.h5'")

# Step 8: Convert the Model to TensorFlow Lite Format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('calorie_prediction_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as 'calorie_prediction_model.tflite'")

from sklearn.metrics import r2_score
from tensorflow.keras.losses import MeanAbsoluteError

# Step 9: Evaluate the Model
y_pred = model.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

# Optional: Calculate Mean Absolute Error (MAE)
mae_fn = MeanAbsoluteError()
mae = mae_fn(y_test, y_pred).numpy()
print(f"Mean Absolute Error: {mae}")

# Step 10: Calculate Accuracy (within a tolerance)
tolerance = 0.1  # 10% tolerance
within_tolerance = abs((y_pred.flatten() - y_test) / y_test) <= tolerance
accuracy = within_tolerance.mean() * 100  # Convert to percentage
print(f"Accuracy (within 10% tolerance): {accuracy:.2f}%")
