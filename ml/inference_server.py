"""
Inference server for mammogram classification.
Run: python ml/inference_server.py
Then page 4 will call http://localhost:5000/classify
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
import json
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={
    r"/classify": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500"]
    },
    r"/health": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500"]
    }
})

# Load model and metadata
MODEL_PATH = Path(__file__).parent.parent / 'artifacts' / 'mammogram_classifier.keras'
CLASS_MAP_PATH = Path(__file__).parent.parent / 'artifacts' / 'class_map.json'

print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    model = None

if CLASS_MAP_PATH.exists():
    with open(CLASS_MAP_PATH, 'r') as f:
        class_map = json.load(f)
    print(f"[OK] Class map loaded: {class_map}")
else:
    class_map = {0: 'normal', 1: 'benign', 2: 'malignant'}

IMG_SIZE = 224

def preprocess_image(img):
    """Convert PIL Image to model input."""
    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Convert to array
    arr = np.array(img, dtype=np.float32)
    # Handle grayscale or RGB
    if len(arr.shape) == 2:  # Grayscale
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:  # RGBA
        arr = arr[:, :, :3]
    # Normalize to [0, 1]
    if arr.max() > 1:
        arr = arr / 255.0
    return arr

@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify():
    """Classify mammogram image."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(image_data))
        
        # Preprocess
        arr = preprocess_image(img)
        # Add batch dimension
        arr = np.expand_dims(arr, 0)
        
        # Predict
        predictions = model.predict(arr, verbose=0)
        probs = predictions[0]
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        
        # Get class name
        class_name = class_map.get(int(class_idx), class_map.get(str(class_idx), 'unknown'))
        
        result = {
            'class': class_name,
            'confidence': confidence,
            'probabilities': {
                'normal': float(probs[0]),
                'benign': float(probs[1]),
                'malignant': float(probs[2])
            }
        }
        
        print(f"Classification: {class_name} (confidence: {confidence:.3f})")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Mammogram Classification Inference Server")
    print("="*50)
    print("Starting server on http://127.0.0.1:5000")
    print("Endpoint: POST /classify")
    print("="*50 + "\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
