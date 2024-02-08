import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from flask_cors import CORS  # Import the CORS module
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your custom model
model = load_model('final_model.h5')

def predict_brain_tumour(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    prediction = model.predict(img_array)

    # Choose a threshold (e.g., 0.5) to decide the class
    threshold = 0.5
    predicted_label = "Tumour Detected" if prediction[0][0] > threshold else "No Tumour Detected"

    return predicted_label

@app.route('/predict-brain-tumour', methods=['POST'])
def predict_brain_tumour_api():
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

        # Perform prediction using the existing predict_brain_tumour function
        predicted_label = predict_brain_tumour(filepath)

        # You can customize the response based on your requirements
        return jsonify({'predicted_label': predicted_label})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'predicted_label': 'Error', 'error_message': 'Error detecting brain tumour. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True)
