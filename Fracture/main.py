import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from predictions import predict
from flask_cors import CORS
from pyngrok import ngrok
ngrok.set_auth_token("2azfWAek6kbMbF6kqYAuUHJSdDy_5LawQiFcgnjT6spNtkqU4")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
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
        bone_type_result = predict(filepath)
        result = predict(filepath, bone_type_result)

        # You can customize the response based on your requirements
        return jsonify({'bone_type': bone_type_result, 'result': result})

    return jsonify({'error': 'Invalid file format'})

if __name__ == "__main__":
    # Use Ngrok to tunnel the Flask app to the public internet
    from pyngrok import ngrok

    # Get the dev server port (defaults to 5000 for Flask, can be overridden with `--port`
    # when starting the server
    port = 5000

    # Open a Ngrok tunnel to the dev server
    public_url = ngrok.connect(port)
    print(' * ngrok tunnel "http://127.0.0.1:5000" -> "{}"'.format(public_url))

    # Update any base URLs or webhooks to use the public ngrok URL
    app.config['BASE_URL'] = public_url  # You can use this base URL in your frontend

    # Start the Flask app
    app.run()
