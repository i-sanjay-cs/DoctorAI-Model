from keras.models import load_model
from keras.preprocessing import image
import numpy as np

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

    # Print the predicted label
    print(f"Predicted Label: {predicted_label}")


