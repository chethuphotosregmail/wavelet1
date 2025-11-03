from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

app = Flask(__name__)

ROI_SIZE = 1080  # your trained image size

# ---------------- MODEL LOADING ----------------
if not os.path.exists("model_v5.pkl") or not os.path.exists("scaler_v5.pkl"):
    raise FileNotFoundError("⚠️ Run train_model_v5.py first to generate model_v5.pkl and scaler_v5.pkl")

rf = joblib.load("model_v5.pkl")
scaler = joblib.load("scaler_v5.pkl")
print("✅ Model_v5 & Scaler_v5 loaded successfully!")


# ---------------- NOTE DETECTION ----------------
def detect_note_region(img):
    """
    Automatically detects the main note region using edge & contour detection.
    Crops it and returns a focused, rectangular region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 60, 160)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # fallback to full image if no note detected

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # ignore too small contours
    if w * h < 0.2 * img.shape[0] * img.shape[1]:
        return img

    cropped = img[y:y + h, x:x + w]
    return cropped


# ---------------- PREPROCESSING ----------------
def preprocess_image(path, size=(ROI_SIZE, ROI_SIZE)):
    """
    Read captured image → detect region → crop → convert to grayscale → resize to match training.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid or unreadable image.")

    # Step 1: detect and crop the note region
    note_roi = detect_note_region(img)

    # Step 2: resize to your trained resolution
    note_roi = cv2.resize(note_roi, size)

    # Step 3: convert to grayscale & normalize lighting (same as training)
    gray = cv2.cvtColor(note_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return note_roi, gray


# ---------------- FEATURE EXTRACTION ----------------
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
    feats = wavelet_color_features(img_color) + lbp_texture_features(gray)

    # add clarity + reflectivity metrics for Xerox differentiation
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # sharpness
    reflectivity = np.sum(gray > 230) / gray.size    # highlights
    contrast = np.std(gray)                          # contrast level

    feats.extend([lap_var, reflectivity, contrast])
    return np.array(feats)


# ---------------- FLASK ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("result.html", result="⚠️ No image selected!")

        # save temporary uploaded image
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # preprocess (crop + gray + normalize)
        img_color, gray = preprocess_image(temp_path)

        # extract + scale + predict
        feats = extract_features(img_color, gray).reshape(1, -1)
        feats = scaler.transform(feats)
        proba = rf.predict_proba(feats)[0]
        genuine_prob = float(proba[1])

        # reflection and texture post-check
        reflectivity = np.sum(gray > 230) / gray.size
        contrast = np.std(gray)

        if genuine_prob >= 0.9 and reflectivity < 0.001 and contrast < 25:
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


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
