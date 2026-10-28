# Plant Disease Detection AI (PlantGuard AI)

An end-to-end Computer Vision and Deep Learning web application designed to classify plant leaf diseases from images and recommend targeted medicines and treatments. The system detects whether a plant leaf is **Healthy**, infected with **Powdery Mildew**, or infected with **Rust**.

Built with **Python**, **Flask**, **TensorFlow / Keras**, and optimized for macOS (Apple Silicon M-series & Intel) and cross-platform environments.

---

## What's New

1. **Modern UI/UX (PlantGuard AI):**
   - Clean glassmorphism interface with responsive cards and smooth animations.
   - Drag-and-drop file upload with instant image preview and removal.
   - Animated plant-leaf loading screen while inference runs.
   - Interactive multi-class confidence meters and disease severity badges.
2. **Live Webcam Capture:**
   - Real-time video preview using your laptop or device camera.
   - Instant frame capture with retake option.
   - Seamless base64 transmission to the backend inference engine.
3. **Medicine & Treatment Recommendations:**
   - **Chemical Treatments:** Specific fungicides, active ingredients (e.g. Sulfur, Mancozeb, Chlorothalonil, Propiconazole), dosage notes, and application frequencies.
   - **Organic & Home Remedies:** Eco-friendly recipes (baking soda, neem oil spray, milk solution, garlic extract).
   - **Cultural & Preventive Practices:** Pruning, spacing, base-watering, sanitization, and soil management tips.

---

## Project Structure

```text
Plant_Disease_Detection_AI/
├── app.py                  # Main Flask web application, APIs & medicine logic
├── model.h5                # CNN trained model weights (HDF5 format)
├── train.py                # Standalone training script for retraining model.h5
├── test_app.py             # Automated unit & integration test suite (12 tests)
├── Model_Training.ipynb    # Jupyter Notebook for data exploration & CNN training
├── requirements.txt        # Verified Python dependencies
├── README.md               # Setup and usage documentation
├── templates/
│   ├── index.html          # Modern PlantGuard AI single-page interface
│   └── import.html         # Base layout with navbar, fonts, and scripts
├── static/
│   ├── css/
│   │   ├── style_v2.css    # Modern glassmorphism UI styles
│   │   ├── bootstrap.min.css
│   │   └── test.css
│   └── js/
│       ├── app.js          # Mode switcher, drag-drop, webcam & results renderer
│       ├── newjs.js        # Legacy upload handler
│       └── bootstrap.min.js
├── uploads/                # Temporary storage for uploaded leaf images
└── Dataset/                # (Optional) Dataset directory for training (Train, Validation, Test)
```

---

## System Requirements

- **Operating System:** macOS (Apple Silicon M1/M2/M3/M4 or Intel), Linux, or Windows
- **Python:** 3.10, 3.11, 3.12, or 3.13
- **Hardware Acceleration:** Native Apple Silicon (`arm64`) Metal/NEON support via TensorFlow 2.16+
- **Browser:** Any modern browser supporting WebRTC (Chrome, Safari, Firefox, Edge)

---

## Quick Start on macOS (Apple Silicon)

Follow these terminal commands in order:

### 1. Navigate to the project directory
```bash
cd /Users/monikar/Downloads/Monika/Plant_Disease_Detection_AI
```

### 2. Create and activate a Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip and install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run automated tests to verify the setup
```bash
python test_app.py
```
*All 12 tests will run and pass cleanly (`OK`).*

### 5. Launch the web application
```bash
python app.py
```

### 6. Access the application in your browser
Open your browser and navigate to:
```text
http://127.0.0.1:5001/
```

> [!NOTE]
> **Why port 5001 on macOS?**  
> On macOS Monterey and later, Apple's **AirPlay Receiver** (`ControlCenter`) listens on port 5000 by default. `app.py` is configured to run on port 5001 to prevent `[Errno 48] Address already in use` conflicts. You can specify a custom port at any time via:
> ```bash
> PORT=5000 python app.py
> ```

---

## Application Features & Usage

### Option A: Upload an Image
1. Click the **Upload Image** tab.
2. Drag and drop a plant leaf image into the drop zone, or click **Choose File**.
3. Click **Analyze Leaf**.
4. View the diagnosis, confidence breakdown, severity, and tailored treatment recommendations.

### Option B: Live Webcam Capture
1. Click the **Live Camera** tab.
2. Click **Start Camera** and grant browser camera permissions.
3. Align the plant leaf in front of the lens and click **Capture**.
4. Click **Analyze Leaf** (or click the retake button if you want to capture again).

---

## API Endpoints

- `GET /` — Renders the web interface.
- `POST /predict` — Accepts multipart form data with image file `file`. Returns JSON diagnosis, confidence, and treatment recommendations.
- `POST /predict_webcam` — Accepts JSON `{"image": "data:image/jpeg;base64,..."}`. Returns JSON diagnosis, confidence, and treatment recommendations.

---

## Model Information

- **Architecture:** Convolutional Neural Network (CNN)
  - Input Layer: `(225, 225, 3)` RGB images
  - Conv2D (32 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2)
  - Flatten + Dense (64 units, ReLU)
  - Output Dense (3 units, Softmax)
- **Classes:**
  - `0`: Healthy
  - `1`: Powdery (Powdery Mildew)
  - `2`: Rust

### Model Training & Retraining

1. **Dataset:** The model is built for the **Plant Disease Recognition Dataset** (approx. 1,530 images split into Train, Validation, and Test sets).
   - Source on Kaggle: [rashikrahmanpritom/plant-disease-recognition-dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)
2. **Directory Structure:** Download and extract the dataset into a `Dataset` folder:
   ```text
   Dataset/
   ├── Train/
   │   ├── Healthy/
   │   ├── Powdery/
   │   └── Rust/
   ├── Validation/
   │   ├── Healthy/
   │   ├── Powdery/
   │   └── Rust/
   └── Test/
       ├── Healthy/
       ├── Powdery/
       └── Rust/
   ```
3. **Train via command line:**
   ```bash
   python train.py
   ```
   Or train interactively via the Jupyter Notebook:
   ```bash
   jupyter notebook Model_Training.ipynb
   ```
   The training pipeline runs for 5 epochs with data augmentation and automatically saves the updated weights to `model.h5`.

---

## Testing

Run the automated test suite anytime using:
```bash
python test_app.py
```

The test suite covers:
- Verification of `model.h5` existence and loading.
- Model input shape `(None, 225, 225, 3)` and output shape `(None, 3)`.
- Label index mapping verification.
- Complete disease info and recommendation structure validation.
- HTTP `GET /` interface rendering.
- HTTP `POST /predict` file validation and error handling.
- HTTP `POST /predict_webcam` base64 image decoding and inference.
- End-to-end leaf disease prediction with sample test images.
- Image preprocessing normalization range `[0, 1]`.
