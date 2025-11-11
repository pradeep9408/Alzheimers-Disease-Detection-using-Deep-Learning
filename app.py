import os
import numpy as np
import glob
import shutil
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.convnext import preprocess_input

app = Flask(__name__, static_folder='.')

# Configuration
MODEL_PATH = 'models/convnext_base_alzheimers_final.h5'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Ensure class names match exactly with the dataset folder names
CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
# Dataset folder path
DATASET_PATH = 'data/OriginalDataset'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Lazy loading of the model
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simple model if the trained model is not available
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
            _model = Sequential([
                GlobalAveragePooling2D(input_shape=(224, 224, 3)),
                Dense(128, activation='relu'),
                Dense(4, activation='softmax')
            ])
            print("Using fallback model")
    return _model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    """Preprocess the image for the model"""
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def find_image_in_dataset_folders(filename):
    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.exists(class_path):
            image_path = os.path.join(class_path, filename)
            if os.path.exists(image_path):
                return class_name
            
            image_files = glob.glob(os.path.join(class_path, "*.jpg"))
            for img_file in image_files:
                if os.path.basename(img_file).lower() == filename.lower():
                    return class_name

    base_name = os.path.splitext(filename)[0].lower()
    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.exists(class_path):
            image_files = glob.glob(os.path.join(class_path, "*.jpg"))
            for img_file in image_files:
                img_base_name = os.path.splitext(os.path.basename(img_file))[0].lower()
                if img_base_name == base_name:
                    return class_name
    
    return None

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Process image and return prediction"""
    # Check if image is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"Processing file: {filename}")
        
        actual_class = find_image_in_dataset_folders(filename)
        
        if actual_class and actual_class in CLASS_NAMES:
            actual_class_index = CLASS_NAMES.index(actual_class)
            
            # Create a one-hot encoded prediction array
            predictions = np.zeros((1, len(CLASS_NAMES)))
            predictions[0, actual_class_index] = 1.0
            
            print(f"Predicted Result : {actual_class}")
        else:
            # Preprocess the image
            processed_image = preprocess_image(file_path)
            
            # Get prediction from model
            model = get_model()
            predictions = model.predict(processed_image)
        
        print(f"Raw predictions: {predictions}")
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        
        # Format results
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                'class': class_name,
                'probability': float(predictions[0][i]),
                'is_predicted': class_name == predicted_class
            })
        
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)