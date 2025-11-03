from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# ---------------- CONFIG ----------------
ROI_SIZE = 1080
DEBUG_SAVE = False  # if True, saves cropped grayscale image for inspection

# ---------------- MODEL LOADING ----------------
if not os.path.exists("model_v5.pkl") or not os.path.exists("scaler_v5.pkl"):
    raise FileNotFoundError("⚠️ Run train.py first to generate model_v5.pkl and scaler_v5.pkl")

rf = joblib.load("model_v5.pkl")
scaler = joblib.load("scaler_v5.pkl")
print("✅ Model_v5 & Scaler_v5 loaded successfully!")


# ---------------- 1. DETECT NOTE REGION ----------------
def detect_note_region(img):
    """Detect and crop main rectangular note region."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < 0.2 * img.shape[0] * img.shape[1]:
        return img
    cropped = img[y:y+h, x:x+w]
    return cropped


# ---------------- 2. PREPROCESS ----------------
def preprocess_image(path, size=(ROI_SIZE, ROI_SIZE)):
    """Crop, resize, and grayscale image exactly like training."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unreadable or invalid image.")

    img = detect_note_region(img)
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if DEBUG_SAVE:
        cv2.imwrite("debug_cropped_gray.png", gray)

    return img, gray


# ---------------- 3. FEATURE EXTRACTION ----------------
def wavelet_color_features(img, wavelet_name='db2'):
    feats = []
    for channel in cv2.split(img):
        coeffs2 = pywt.dwt2(channel, wavelet_name)
        cA, (cH, cV, cD) = coeffs2
        for mat in [cH, cV, cD]:
            hist, _ = np.histogram(mat.flatten(), bins=64, density=True)
            feats.extend([np.var(hist), skew(hist), kurtosis(hist)])
    return feats


def lbp_texture_features(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), density=True)
    return hist.tolist()


def extract_features(img_color, gray):
    """Exact replica of your training feature extraction."""
    feats = wavelet_color_features(img_color) + lbp_texture_features(gray)

    # ---- Sharpness / contrast / reflection / edge metrics ----
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    contrast_map = cv2.absdiff(gray, blur)
    contrast_std = np.std(contrast_map)
    bright_ratio = np.sum(gray > 220) / gray.size

    bright_mask = cv2.inRange(gray, 230, 255)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reflection_clusters = len([c for c in contours if 5 < cv2.contourArea(c) < 200])
    reflection_density = reflection_clusters / (gray.shape[0] * gray.shape[1] / 10000)

    color_std = np.std(cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)[:, :, 0])
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_uniformity = 1 / (np.std(edge_mag) + 1e-5)
    edges = cv2.Canny(gray, 80, 160)
    micro_edge_density = np.sum(edges > 0) / edges.size
    specular_ratio = np.sum(gray > 245) / gray.size
    highpass = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    texture_coarseness = np.std(highpass)

    feats.extend([
        lap_var, contrast_std, bright_ratio, reflection_density,
        color_std, edge_uniformity, micro_edge_density,
        specular_ratio, texture_coarseness
    ])

    return np.array(feats, dtype=np.float32)


# ---------------- 4. FLASK ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("result.html", result="⚠️ No image selected!")

        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # Preprocess and extract
        img_color, gray = preprocess_image(temp_path)
        feats = extract_features(img_color, gray).reshape(1, -1)

        # Scale + predict
        feats_scaled = scaler.transform(feats)
        proba = rf.predict_proba(feats_scaled)[0]
        genuine_prob = float(proba[1])

        # Reflection + texture validation
        reflectivity = np.sum(gray > 230) / gray.size
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Decision logic
        if genuine_prob >= 0.9 and reflectivity < 0.001 and texture_var < 40:
            result = "⚠️ Possible Xerox Copy"
        elif genuine_prob >= 0.9:
            result = "✅ Genuine Note"
        elif genuine_prob <= 0.6:
            result = "❌ Fake / Xerox Note"
        else:
            result = "⚠️ Suspicious Note"

        confidence = f"{genuine_prob:.2f}"

        os.remove(temp_path)
        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")


# ---------------- 5. RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
