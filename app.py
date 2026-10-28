import os
import io
import base64
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# ─── Disease Information & Treatment Recommendations ───────────────────────
DISEASE_INFO = {
    'Healthy': {
        'status': 'healthy',
        'icon': '✅',
        'description': 'Your plant leaf appears healthy with no visible signs of disease. '
                       'The leaf shows normal color, texture, and growth patterns.',
        'severity': 'None',
        'recommendations': [
            {
                'type': 'Preventive Care',
                'icon': '🛡️',
                'items': [
                    'Continue regular watering schedule — avoid overwatering',
                    'Ensure proper air circulation around plants',
                    'Apply balanced NPK fertilizer every 2–4 weeks',
                    'Monitor regularly for early signs of disease',
                ]
            },
            {
                'type': 'Best Practices',
                'icon': '🌱',
                'items': [
                    'Rotate crops seasonally to prevent soil-borne diseases',
                    'Remove fallen leaves and debris from the base',
                    'Use mulch to retain moisture and suppress weeds',
                    'Inspect neighboring plants to catch infections early',
                ]
            }
        ]
    },
    'Powdery': {
        'status': 'diseased',
        'icon': '⚠️',
        'description': 'Powdery Mildew detected — a fungal disease caused by various species of '
                       'the order Erysiphales. It appears as white or grayish powdery spots on '
                       'leaves and stems, eventually causing yellowing, wilting, and premature leaf drop.',
        'severity': 'Moderate',
        'recommendations': [
            {
                'type': 'Chemical Treatment',
                'icon': '🧪',
                'items': [
                    '<strong>Sulfur-based fungicide</strong> — Apply wettable sulfur spray '
                    '(e.g., Bonide Sulfur Plant Fungicide) every 7–14 days',
                    '<strong>Neem Oil</strong> — Spray 2–3 tablespoons per gallon of water; '
                    'apply weekly as both treatment and preventive',
                    '<strong>Potassium bicarbonate</strong> — Mix 1 tbsp per gallon of water '
                    'with a few drops of liquid soap; spray on affected areas',
                    '<strong>Myclobutanil fungicide</strong> (e.g., Spectracide Immunox) — '
                    'Apply as directed for severe infections',
                ]
            },
            {
                'type': 'Organic / Home Remedies',
                'icon': '🌿',
                'items': [
                    '<strong>Baking soda solution</strong> — 1 tbsp baking soda + ½ tsp liquid '
                    'soap per gallon of water; spray weekly',
                    '<strong>Milk spray</strong> — Mix 1 part milk to 9 parts water; spray '
                    'every 10 days (lactic acid combats fungal growth)',
                    '<strong>Garlic extract spray</strong> — Blend 2 garlic bulbs in 1 quart '
                    'of water, strain, dilute 1:10, and spray on leaves',
                    '<strong>Apple cider vinegar</strong> — 2–3 tbsp per gallon of water; '
                    'spray in early morning',
                ]
            },
            {
                'type': 'Cultural Practices',
                'icon': '🌾',
                'items': [
                    'Remove and destroy all infected leaves immediately',
                    'Improve air circulation — space plants adequately and prune dense foliage',
                    'Water at the base of plants (avoid wetting leaves)',
                    'Avoid excessive nitrogen fertilizers (promotes tender growth susceptible to mildew)',
                    'Plant resistant cultivars in future growing seasons',
                ]
            }
        ]
    },
    'Rust': {
        'status': 'diseased',
        'icon': '🔴',
        'description': 'Rust disease detected — caused by fungi in the order Pucciniales. '
                       'Characterized by orange, yellow, or reddish-brown pustules (spore masses) '
                       'on the undersides of leaves. Can cause severe defoliation and weaken the plant.',
        'severity': 'High',
        'recommendations': [
            {
                'type': 'Chemical Treatment',
                'icon': '🧪',
                'items': [
                    '<strong>Mancozeb fungicide</strong> — Apply as a protective spray every '
                    '7–10 days (e.g., Dithane M-45)',
                    '<strong>Propiconazole</strong> (e.g., Banner Maxx, Tilt) — Systemic '
                    'fungicide; apply at first sign of rust pustules',
                    '<strong>Chlorothalonil</strong> (e.g., Daconil) — Broad-spectrum '
                    'protectant; spray every 7–14 days',
                    '<strong>Copper-based fungicide</strong> (e.g., Bordeaux mixture) — '
                    'Apply early in the season as a preventive measure',
                    '<strong>Tebuconazole</strong> (e.g., Folicur) — Effective systemic '
                    'treatment for advanced rust infections',
                ]
            },
            {
                'type': 'Organic / Home Remedies',
                'icon': '🌿',
                'items': [
                    '<strong>Neem oil spray</strong> — 2 tbsp per gallon of water; '
                    'apply every 7 days on both leaf surfaces',
                    '<strong>Baking soda + oil spray</strong> — 1 tbsp baking soda + '
                    '1 tbsp vegetable oil + few drops dish soap per gallon of water',
                    '<strong>Sulfur dust</strong> — Apply dusting sulfur directly to '
                    'affected leaves in dry conditions',
                    '<strong>Compost tea</strong> — Brew aerated compost tea and spray '
                    'on foliage to introduce beneficial microorganisms',
                ]
            },
            {
                'type': 'Cultural Practices',
                'icon': '🌾',
                'items': [
                    'Remove and <strong>burn</strong> (do not compost) all infected plant material',
                    'Avoid overhead irrigation — water at soil level only',
                    'Ensure wide spacing between plants for air circulation',
                    'Apply mulch to prevent rain-splashed spores from reaching leaves',
                    'Rotate crops — avoid planting the same species in the same spot for 2–3 years',
                    'Disinfect pruning tools with 10% bleach solution between cuts',
                ]
            }
        ]
    }
}


def getResult(image_path):
    """Preprocess image and return model prediction probabilities."""
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


def process_image_array(img_array):
    """Process a numpy image array (from webcam) for prediction."""
    img = Image.fromarray(img_array)
    img = img.resize((225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


# ─── Load Model ────────────────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from '{MODEL_PATH}'.")
    except Exception as e:
        print(f"Error loading model from '{MODEL_PATH}': {e}")
else:
    print(f"Warning: Model file not found at '{MODEL_PATH}'.")
    print("Please generate or train 'model.h5' to enable predictions.")


# ─── Routes ────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if model is None:
        return jsonify({'error': 'Model not loaded. Ensure model.h5 exists.'}), 500

    filename = secure_filename(f.filename)
    if not filename:
        filename = "upload.jpg"
    file_path = os.path.join(UPLOADS_DIR, filename)
    f.save(file_path)

    predictions = getResult(file_path)
    idx = int(np.argmax(predictions))
    predicted_label = labels[idx]
    confidence = float(predictions[idx]) * 100

    result = {
        'label': predicted_label,
        'confidence': round(confidence, 1),
        'probabilities': {labels[i]: round(float(predictions[i]) * 100, 1) for i in range(3)},
        'info': DISEASE_INFO[predicted_label]
    }
    return jsonify(result)


@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    """Accept a base64-encoded webcam frame and return a prediction."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400
    if model is None:
        return jsonify({'error': 'Model not loaded. Ensure model.h5 exists.'}), 500

    try:
        # Decode the base64 image
        img_data = data['image']
        # Remove the data:image/...;base64, prefix if present
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)

        # Save the captured frame
        file_path = os.path.join(UPLOADS_DIR, 'webcam_capture.jpg')
        img.save(file_path)

        predictions = process_image_array(img_array)
        idx = int(np.argmax(predictions))
        predicted_label = labels[idx]
        confidence = float(predictions[idx]) * 100

        result = {
            'label': predicted_label,
            'confidence': round(confidence, 1),
            'probabilities': {labels[i]: round(float(predictions[i]) * 100, 1) for i in range(3)},
            'info': DISEASE_INFO[predicted_label]
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Failed to process webcam image: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"Server starting on http://127.0.0.1:{port}/")
    app.run(debug=True, host='127.0.0.1', port=port)
