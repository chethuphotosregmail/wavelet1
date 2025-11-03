from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, hog
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

ROI_SIZE = 1080
GENUINE_REF_DIR = "resolution/genuine"

# ---------------- MODEL LOAD ----------------
rf = joblib.load("model_v5.pkl")
scaler = joblib.load("scaler_v5.pkl")
print("✅ Model_v5 & Scaler_v5 loaded successfully!")

# ---------------- HOG REFERENCE ----------------
def compute_reference_hog():
    hogs = []
    for fname in os.listdir(GENUINE_REF_DIR):
        fpath = os.path.join(GENUINE_REF_DIR, fname)
        if not fpath.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        h = hog(img, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        hogs.append(h)
    if len(hogs) == 0:
        return None
    return np.mean(hogs, axis=0)

REF_HOG = compute_reference_hog()

# ---------------- PREPROCESS ----------------
def preprocess_image(path, size=(ROI_SIZE, ROI_SIZE)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unreadable image.")
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # normalize lighting using cumulative histogram
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = 255 * cdf / cdf[-1]
    gray = np.interp(gray.flatten(), bins[:-1], cdf_normalized).reshape(gray.shape).astype(np.uint8)
    return img, gray

# ---------------- FEATURE EXTRACTION ----------------
def wavelet_color_features(img, wavelet_name='db2'):
    feats = []
    for channel in cv2.split(img):
        coeffs2 = pywt.dwt2(channel, wavelet_name)
        _, (cH, cV, cD) = coeffs2
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
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast_std = np.std(cv2.absdiff(gray, cv2.GaussianBlur(gray, (9, 9), 0)))
    bright_ratio = np.sum(gray > 220) / gray.size
    feats.extend([lap_var, contrast_std, bright_ratio])
    return np.array(feats, dtype=np.float32)

# ---------------- DETECT POLARITY ----------------
def check_polarity(gray):
    """Detect if lighting or tone inversion is present."""
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)
    high_ratio = np.sum(gray > 200) / gray.size
    low_ratio = np.sum(gray < 50) / gray.size
    return (mean_intensity < 110 and high_ratio < 0.05 and low_ratio > 0.3)

# ---------------- EXTRA METRICS ----------------
def micro_entropy(gray):
    patch_std = []
    for i in range(0, gray.shape[0], 40):
        for j in range(0, gray.shape[1], 40):
            patch = gray[i:i+40, j:j+40]
            if patch.size > 0:
                patch_std.append(np.std(patch))
    return np.mean(patch_std)

# ---------------- PREDICT ----------------
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
        img_color, gray = preprocess_image(temp_path)

        # detect inverted tone & fix
        if check_polarity(gray):
            gray = cv2.bitwise_not(gray)

        feats = extract_features(img_color, gray).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        proba = rf.predict_proba(feats_scaled)[0]
        genuine_prob = float(proba[1])

        test_hog = hog(cv2.resize(gray, (256, 256)), orientations=9,
                       pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                       visualize=False, block_norm='L2-Hys')
        pattern_similarity = cosine_similarity([REF_HOG], [test_hog])[0][0] if REF_HOG is not None else 0.0
        entropy_val = micro_entropy(gray)

        # fixed thresholding
        if genuine_prob > 0.85 and entropy_val > 16 and pattern_similarity > 0.8:
            result = "✅ Genuine Note"
        elif genuine_prob > 0.7 and pattern_similarity > 0.75:
            result = "⚠️ Likely Genuine (Low Light)"
        else:
            result = "❌ Fake / Xerox Detected"

        confidence = f"Model: {genuine_prob:.2f} | Pattern: {pattern_similarity:.2f} | Entropy: {entropy_val:.2f}"

        os.remove(temp_path)
        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
