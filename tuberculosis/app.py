import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
from flask_cors import CORS
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# Define the path to your saved model
model_path = 'tb3_model.h5'

# Load the saved model
model = load_model(model_path)

def predict_tuberculosis(image_path, model):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction using the model
    prediction = model.predict(img_array)

    # Calculate confidence of the predicted class
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    confidence_percentage = confidence * 100

    # Determine predicted class based on confidence
    predicted_class = "Tuberculosis Detected" if prediction[0][0] > 0.5 else "Normal(No Tuberculosis  Detected)"

    return predicted_class, confidence_percentage

@app.route('/predict-tuberculosis', methods=['POST'])
def predict_tuberculosis_api():
    try:
        # Check if the 'file' is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if no file was selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Create the 'uploads' directory if it does not exist
        os.makedirs('uploads', exist_ok=True)

        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Perform prediction using the predict_tuberculosis function
        predicted_class, confidence_percentage = predict_tuberculosis(filepath, model)

        # You can customize the response based on your requirements
        return jsonify({'predicted_class': predicted_class, 'confidence_percentage': confidence_percentage})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'predicted_class': 'Error', 'confidence_percentage': 0,
                        'error_message': 'Error detecting tuberculosis. Please try again.'})

if __name__ == '__main__':
    # Open a Ngrok tunnel to the Flask app
    public_url = ngrok.connect(5000)  # Port 5000 is where Flask app runs
    print(' * ngrok tunnel "http://127.0.0.1:5000" -> "{}"'.format(public_url))

    # Run the Flask app
    app.run()
