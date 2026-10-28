#!/usr/bin/env python3
"""
Test suite for Plant Disease Detection AI.
Verifies model architecture, image preprocessing, Flask web endpoints,
webcam endpoint, and disease recommendation data.
"""

import os
import json
import base64
import unittest
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from app import app, getResult, labels, MODEL_PATH, UPLOADS_DIR, DISEASE_INFO


class PlantDiseaseDetectionTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # ── Model Tests ──────────────────────────────────────────
    def test_model_file_exists(self):
        """Verify model.h5 exists and can be loaded."""
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        self.assertIsNotNone(model)

    def test_model_input_and_output_shapes(self):
        """Verify the CNN model input shape is (None, 225, 225, 3) and output has 3 classes."""
        model = load_model(MODEL_PATH)
        self.assertEqual(model.input_shape, (None, 225, 225, 3))
        self.assertEqual(model.output_shape, (None, 3))

    def test_labels_mapping(self):
        """Verify the 3 expected labels."""
        self.assertEqual(labels, {0: 'Healthy', 1: 'Powdery', 2: 'Rust'})

    # ── Disease Info Tests ───────────────────────────────────
    def test_disease_info_keys(self):
        """Verify DISEASE_INFO has entries for all 3 labels."""
        for label in ['Healthy', 'Powdery', 'Rust']:
            self.assertIn(label, DISEASE_INFO)

    def test_disease_info_has_recommendations(self):
        """Verify each disease entry has non-empty recommendations."""
        for label, info in DISEASE_INFO.items():
            self.assertIn('recommendations', info)
            self.assertTrue(len(info['recommendations']) > 0,
                            f"No recommendations for {label}")
            for rec in info['recommendations']:
                self.assertIn('type', rec)
                self.assertIn('items', rec)
                self.assertTrue(len(rec['items']) > 0)

    def test_disease_info_structure(self):
        """Verify each disease entry has required fields."""
        required_fields = ['status', 'icon', 'description', 'severity', 'recommendations']
        for label, info in DISEASE_INFO.items():
            for field in required_fields:
                self.assertIn(field, info, f"Missing '{field}' in DISEASE_INFO['{label}']")

    # ── Route Tests ──────────────────────────────────────────
    def test_index_route(self):
        """Verify GET / renders the main HTML interface."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'PlantGuard', response.data)

    def test_predict_without_file(self):
        """Verify POST /predict without file returns 400."""
        response = self.app.post('/predict')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_predict_with_sample_image(self):
        """Verify POST /predict returns JSON with label, confidence, and info."""
        sample_images = [f for f in os.listdir(UPLOADS_DIR)
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.assertTrue(len(sample_images) > 0, "No sample images found")
        sample_path = os.path.join(UPLOADS_DIR, sample_images[0])

        with open(sample_path, 'rb') as img_file:
            data = {'file': (img_file, sample_images[0])}
            response = self.app.post('/predict', data=data,
                                     content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn(result['label'], ['Healthy', 'Powdery', 'Rust'])
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
        self.assertIn('info', result)
        self.assertIn('recommendations', result['info'])

    def test_predict_webcam_endpoint(self):
        """Verify POST /predict_webcam returns valid prediction for a base64 image."""
        # Create a simple test image
        img = Image.new('RGB', (225, 225), color=(100, 200, 100))
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()

        response = self.app.post('/predict_webcam',
                                 data=json.dumps({'image': img_base64}),
                                 content_type='application/json')

        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn(result['label'], ['Healthy', 'Powdery', 'Rust'])
        self.assertIn('info', result)

    def test_predict_webcam_no_image(self):
        """Verify POST /predict_webcam without image data returns 400."""
        response = self.app.post('/predict_webcam',
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)

    # ── Preprocessing Tests ──────────────────────────────────
    def test_image_preprocessing(self):
        """Verify image preprocessing normalizes pixels to [0, 1] range."""
        sample_images = [f for f in os.listdir(UPLOADS_DIR)
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        sample_path = os.path.join(UPLOADS_DIR, sample_images[0])

        img = load_img(sample_path, target_size=(225, 225))
        x = img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)

        self.assertEqual(x.shape, (1, 225, 225, 3))
        self.assertTrue(0.0 <= x.min() <= 1.0)
        self.assertTrue(0.0 <= x.max() <= 1.0)


if __name__ == '__main__':
    unittest.main()
