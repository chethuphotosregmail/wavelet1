from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# -------------------- CONFIG --------------------
ROI_SIZE = 1080
TRAIN_DATA_DIR = "resolution"
MATCH_THRESHOLD = 0.45
DEBUG_SAVE = False

# -------------------- LOAD REFERENCE TEMPLATE --------------------
def load_reference_template():
    """
    Automatically loads one genuine training image from resolution/genuine/
    and uses it as a pattern reference for automatic cropping.
    """
    genuine_folder = os.path.join(TRAIN_DATA_DIR, "genuine")
    if not os.path.exists(genuine_folder):
        raise FileNotFoundError("⚠️ 'resolution/genuine/' folder not found!")

    for fname in os.listdir(genuine_folder):
        if fname.lower().endswith(".png"):
            path = os.path.join(genuine_folder, fname)
            tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tpl is not None:
                print(f"✅ Loaded reference template from: {fname}")
                return tpl

    raise FileNotFoundError("⚠️ No genuine images found inside 'resolution/genuine/' folder!")

# Load reference image automatically
REF_TEMPLATE = load_reference_template()

# -------------------- LOAD MODEL & SCALER --------------------
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    raise FileNotFoundError("⚠️ Run train_model.py first to generate model.pkl and scaler.pkl")

rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------- FEATURE EXTRACTION --------------------
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
    return np.array(wavelet_color_features(img_color) + lbp_texture_features(gray))

# -------------------- ROI DETECTION --------------------
def detect_pattern_region(img):
    """
    Detect the region in the full captured image that matches your genuine pattern.
    Automatically crops it and resizes to 1080x1080.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = REF_TEMPLATE

    scales = [0.5, 0.75, 1.0, 1.25]
    best_val, best_loc, best_size = -1, None, None

    for s in scales:
        tpl = cv2.resize(template, (int(template.shape[1]*s), int(template.shape[0]*s)))
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        if maxVal > best_val:
            best_val = maxVal
            best_loc = maxLoc
            best_size = tpl.shape[::-1]  # (width, height)

    if best_val < MATCH_THRESHOLD:
        raise ValueError("Pattern not found clearly — try closer or clearer photo")

    tw, th = best_size
    x, y = best_loc
    cx, cy = x + tw // 2, y + th // 2

    pad = int(max(tw, th) * 0.5)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(img.shape[1], cx + pad), min(img.shape[0], cy + pad)
    crop = img[y1:y2, x1:x2].copy()

    crop_resized = cv2.resize(crop, (ROI_SIZE, ROI_SIZE))
    if DEBUG_SAVE:
        cv2.imwrite("debug_crop.png", crop_resized)
    return crop_resized

# -------------------- FLASK ROUTES --------------------
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
        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError("Invalid image.")

        roi = detect_pattern_region(img)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        feats = extract_features(roi, gray).reshape(1, -1)
        feats = scaler.transform(feats)
        proba = rf.predict_proba(feats)[0]
        genuine_prob = float(proba[1])

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
