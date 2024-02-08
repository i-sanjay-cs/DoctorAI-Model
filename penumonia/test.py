from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

saved_model_path='pneumonia_detection_with_classes_model.h5'

def predict_pneumonia(img_path):
    # Load the saved model
    loaded_model = load_model(saved_model_path)

    # Load an image for prediction
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocess the image for prediction
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # If class labels are not provided, use default

    class_labels = ['BACTERIAL PNEUMONIA', 'NORMAL', 'VIRUS PNEUMONIA']

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class name
    predicted_class_name = class_labels[predicted_class_index]

    # Get the confidence (percentage) for the predicted class
    confidence_percentage = predictions[0][predicted_class_index] * 100

    return predicted_class_name, confidence_percentage
