from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('my_flower_model.keras')

# Mapping of class indices to flower names
class_names =  ['daisy', 'dandelion','rose', 'sunflower',  'tulip']

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Route to serve files from the 'uploads' folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image temporarily
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get the prediction from the model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        # Return the prediction and image URL
        return jsonify({'predicted_class': predicted_label, 'image_url': f'/uploads/{file.filename}'})

if __name__ == '__main__':
    app.run(debug=True)
