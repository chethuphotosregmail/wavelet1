from flask import Flask, render_template, request  # type: ignore
import cv2, os, tempfile, numpy as np, pywt, joblib  # type: ignore
from scipy.stats import skew, kurtosis  # type: ignore
from skimage.feature import local_binary_pattern  # type: ignore

app = Flask(__name__)

# -------------------- CONFIG --------------------
ROI_SIZE = 1080
DEBUG_SAVE = False  # set to True to save cropped ROI image for inspection

# -------------------- MODEL LOADING --------------------
if not os.path.exists("model_v3.pkl") or not os.path.exists("scaler_v3.pkl"):
    raise FileNotFoundError("⚠️ Run train_model_v3.py first to generate model_v3.pkl and scaler_v3.pkl")

rf = joblib.load("model_v3.pkl")
scaler = joblib.load("scaler_v3.pkl")
print("✅ model_v3 & scaler_v3 loaded successfully!")


# -------------------- 1. DETECT NOTE AREA --------------------
def detect_note_region(img):
    """
    Detects the largest rectangular-like contour (note area)
    and crops it tightly before resizing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # fallback if nothing detected

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # ignore tiny contours (likely background)
    if w * h < 0.2 * img.shape[0] * img.shape[1]:
        return img

    cropped = img[y:y + h, x:x + w]
    return cropped


# -------------------- 2. PREPROCESSING --------------------
def preprocess_image(path, size=(ROI_SIZE, ROI_SIZE)):
    """
    Read image, detect note, crop, convert to grayscale with CLAHE, and resize.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid or unreadable image.")

    note_roi = detect_note_region(img)
    note_roi = cv2.resize(note_roi, size)

    gray = cv2.cvtColor(note_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if DEBUG_SAVE:
        cv2.imwrite("debug_roi.png", note_roi)

    return note_roi, gray


# -------------------- 3. FEATURE EXTRACTION --------------------
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
    """Combine Wavelet, LBP, and clarity-reflectivity features"""
    feats = wavelet_color_features(img_color) + lbp_texture_features(gray)

    # --- clarity / reflectivity / contrast metrics ---
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()              # sharpness
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    contrast_map = cv2.absdiff(gray, blur)
    contrast_std = np.std(contrast_map)                          # local contrast
    bright_ratio = np.sum(gray > 220) / gray.size                # reflective highlights

    feats.extend([lap_var, contrast_std, bright_ratio])
    return np.array(feats)


# -------------------- 4. FLASK ROUTES --------------------
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

        # ---- Preprocess ----
        img_color, gray = preprocess_image(temp_path)

        # ---- Extract & Scale ----
        feats = extract_features(img_color, gray).reshape(1, -1)
        feats = scaler.transform(feats)

        # ---- Predict ----
        proba = rf.predict_proba(feats)[0]
        genuine_prob = float(proba[1])

        # ---- Decision ----
        if genuine_prob >= 0.9:
            result = "✅ Genuine Note"
        elif genuine_prob <= 0.6:
            result = "❌ Fake Note"
        else:
            result = "⚠️ Suspicious Note"

        confidence = f"{genuine_prob:.2f}"

        os.remove(temp_path)
        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")


# -------------------- 5. RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
