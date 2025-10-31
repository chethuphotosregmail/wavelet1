from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

app = Flask(__name__)

# ---------- 1. IMAGE PREPROCESS ----------
def preprocess_image(path, size=(1080, 1080)):
    """
    Reads an image, converts it to grayscale + equalizes lighting,
    resizes to 1080x1080, and returns both color + gray versions.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid or unreadable image.")
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return img, gray


# ---------- 2. FEATURE EXTRACTORS ----------
def wavelet_color_features(img, wavelet_name='db2'):
    """Wavelet statistics extracted separately for each RGB channel"""
    feats = []
    for channel in cv2.split(img):
        coeffs2 = pywt.dwt2(channel, wavelet_name)
        cA, (cH, cV, cD) = coeffs2
        for mat in [cH, cV, cD]:
            hist, _ = np.histogram(mat.flatten(), bins=64, density=True)
            feats.extend([np.var(hist), skew(hist), kurtosis(hist)])
    return feats


def lbp_texture_features(gray):
    """Local Binary Pattern histogram (micro-texture)"""
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), density=True)
    return hist.tolist()


def extract_features(img_color, gray):
    """Combine wavelet + LBP texture features"""
    f1 = wavelet_color_features(img_color)
    f2 = lbp_texture_features(gray)
    return np.array(f1 + f2)


# ---------- 3. LOAD TRAINED MODEL + SCALER ----------
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    raise FileNotFoundError("⚠️ Run train_model.py first to generate model.pkl and scaler.pkl")

print("📂 Loading trained model and scaler...")
rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
print("✅ Model & scaler loaded successfully!")


# ---------- 4. FLASK ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("result.html", result="⚠️ No image selected!")

        # Save uploaded file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # ---- Preprocess Image ----
        img_color, gray = preprocess_image(temp_path)

        # ---- Extract Features ----
        feats = extract_features(img_color, gray).reshape(1, -1)

        # ---- Scale Features ----
        feats = scaler.transform(feats)

        # ---- Predict ----
        proba = rf.predict_proba(feats)[0]
        genuine_prob = float(proba[1])

        # ---- Decision Logic ----
        if genuine_prob >= 0.9:
            result = "✅ Genuine"
        elif genuine_prob <= 0.6:
            result = "❌ Fake"
        else:
            result = "⚠️ Suspicious"

        confidence = f"{genuine_prob:.2f}"

        # ---- Clean up ----
        try:
            os.remove(temp_path)
        except PermissionError:
            pass

        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")


# ---------- 5. RUN THE APP ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
