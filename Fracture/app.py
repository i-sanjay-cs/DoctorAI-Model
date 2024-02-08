# main_app.py

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from predictions import predict

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/predict', methods=['POST'])
def predict_bone_fracture():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform prediction using the predict function from predictions.py
        result = predict(filepath)

        # You can customize the response based on your requirements
        return jsonify({'result': result})

    return jsonify({'error': 'Invalid file format'})

if __name__ == "__main__":
    app.run(debug=True)
