from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# -------------------- CONFIG --------------------
ROI_SIZE = 1080
WINDOW_SIZE_RATIO = 0.25  # portion of image scanned per window (adjustable)
PADDING = 20              # padding around detected ROI
DEBUG_SAVE = False         # True = save cropped image for debug

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

# -------------------- ROI DETECTION (NO TEMPLATE NEEDED) --------------------
def detect_roi_by_texture(img):
    """
    Automatically finds the region of highest texture energy using Laplacian variance.
    Crops that region and resizes to 1080x1080.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    win = max(64, int(min(h, w) * WINDOW_SIZE_RATIO))
    step = max(16, win // 4)

    best_score = -1
    best_box = (0, 0, w, h)

    # Slide window and measure texture (sharpness)
    for y in range(0, h - win + 1, step):
        for x in range(0, w - win + 1, step):
            patch = gray[y:y+win, x:x+win]
            score = cv2.Laplacian(patch, cv2.CV_64F).var()  # Laplacian variance
            if score > best_score:
                best_score = score
                best_box = (x, y, win, win)

    x, y, win, _ = best_box
    cx, cy = x + win // 2, y + win // 2

    # Crop around best region
    pad = int(win * 0.5)
    x1, y1 = max(0, cx - pad), max(0, cy - pad)
    x2, y2 = min(w, cx + pad), min(h, cy + pad)
    roi = img[y1:y2, x1:x2].copy()

    # Resize to 1080x1080
    roi_resized = cv2.resize(roi, (ROI_SIZE, ROI_SIZE))
    if DEBUG_SAVE:
        cv2.imwrite("debug_roi.png", roi_resized)
    return roi_resized

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

        # 1️⃣ Auto-detect region with most detailed pattern
        roi = detect_roi_by_texture(img)

        # 2️⃣ Convert to grayscale + equalize
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 3️⃣ Extract features
        feats = extract_features(roi, gray).reshape(1, -1)
        feats = scaler.transform(feats)

        # 4️⃣ Predict with trained model
        proba = rf.predict_proba(feats)[0]
        genuine_prob = float(proba[1])

        # 5️⃣ Decision
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
