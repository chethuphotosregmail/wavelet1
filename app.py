from flask import Flask, render_template, request
import os, tempfile, cv2, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern

app = Flask(__name__)

ROI_SIZE = 1080

# ---------- load trained model ----------
MODEL_PATH = "model_v5.pkl"
SCALER_PATH = "scaler_v5.pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise FileNotFoundError("Model or scaler file missing.")

rf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ model_v5 & scaler_v5 loaded.")


# ---------- preprocessing ----------
def preprocess_image(path, size=(ROI_SIZE, ROI_SIZE)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unreadable image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img = cv2.resize(img, size)
    gray = cv2.resize(gray, size)
    return img, gray


# ---------- feature extraction (exactly as in train.py) ----------
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

    # same 9 metrics from training script
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
    edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
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


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("result.html", result="⚠️ No image selected!")

        tmp = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(tmp)

        img_color, gray = preprocess_image(tmp)
        feats = extract_features(img_color, gray).reshape(1, -1)
        feats = scaler.transform(feats)
        proba = rf.predict_proba(feats)[0]
        genuine_prob = float(proba[1])

        if genuine_prob >= 0.9:
            result = "✅ Genuine Note"
        elif genuine_prob <= 0.6:
            result = "❌ Fake Note"
        else:
            result = "⚠️ Suspicious Note"

        os.remove(tmp)
        conf = f"{genuine_prob:.2f}"
        return render_template("result.html", result=result, confidence=conf)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
