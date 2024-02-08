import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from flask_cors import CORS
from pyngrok import ngrok  # Import ngrok module for creating a public URL

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model
saved_model_path = 'pneumonia_detection_with_classes_model.h5'
model = load_model(saved_model_path)

# Function to predict pneumonia
def predict_pneumonia(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)

    # Get the class labels
    class_labels = ['BACTERIAL PNEUMONIA', 'NORMAL', 'VIRUS PNEUMONIA']

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class name
    predicted_class_name = class_labels[predicted_class_index]

    # Get the confidence (percentage) for the predicted class
    confidence_percentage = predictions[0][predicted_class_index] * 100

    return predicted_class_name, confidence_percentage

# Endpoint to handle prediction request
@app.route('/predict-pneumonia', methods=['POST'])
def predict_pneumonia_api():
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

        # Perform prediction using the existing predict_pneumonia function
        predicted_class_name, confidence_percentage = predict_pneumonia(filepath)

        # You can customize the response based on your requirements
        return jsonify({'predicted_class_name': predicted_class_name, 'confidence_percentage': confidence_percentage})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'predicted_class_name': 'Error', 'confidence_percentage': 0,
                        'error_message': 'Error detecting pneumonia. Please try again.'})

if __name__ == '__main__':
    # Open a Ngrok tunnel to the Flask app
    public_url = ngrok.connect(5000)  # Port 5000 is where Flask app runs
    print(' * ngrok tunnel "http://127.0.0.1:5000" -> "{}"'.format(public_url))

    # Run the Flask app
    app.run()
