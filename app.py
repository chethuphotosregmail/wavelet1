from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt, joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# -------------------- CONFIG --------------------
ROI_SIZE = 1080
TEMPLATE_FOLDER = "ref_templates"
GENUINE_TEMPLATE = "ref_templates/genuine_ref.png"  # your genuine pattern image here
SSIM_THRESHOLD = 0.75  # below this => fake
CORR_THRESHOLD = 0.70  # below this => fake

# -------------------- PREPROCESS --------------------
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unreadable image.")
    img = cv2.resize(img, (ROI_SIZE, ROI_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return img, gray

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
    f1 = wavelet_color_features(img_color)
    f2 = lbp_texture_features(gray)
    return np.array(f1 + f2)

# -------------------- PATTERN MATCHING FEATURE --------------------
def pattern_similarity_score(gray_img):
    """Compare captured region with reference genuine template."""
    if not os.path.exists(GENUINE_TEMPLATE):
        return 1.0  # skip if template missing (assume genuine)

    ref = cv2.imread(GENUINE_TEMPLATE, cv2.IMREAD_GRAYSCALE)
    ref = cv2.resize(ref, (gray_img.shape[1], gray_img.shape[0]))

    # compute SSIM
    ssim_val = ssim(gray_img, ref)

    # compute normalized cross-correlation
    corr = cv2.matchTemplate(gray_img, ref, cv2.TM_CCOEFF_NORMED).max()

    # combine for final similarity score
    return (ssim_val + corr) / 2.0

# -------------------- LOAD MODEL & SCALER --------------------
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    raise FileNotFoundError("⚠️ Run train_model.py first to generate model.pkl and scaler.pkl")

rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------- ROUTES --------------------
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
        features = extract_features(img_color, gray).reshape(1, -1)
        features = scaler.transform(features)

        proba = rf.predict_proba(features)[0]
        genuine_prob = float(proba[1])

        # calculate pattern similarity
        similarity = pattern_similarity_score(gray)

        # hybrid decision
        if similarity < SSIM_THRESHOLD or genuine_prob < 0.55:
            result = "❌ Fake Note"
        elif genuine_prob > 0.85 and similarity > 0.8:
            result = "✅ Genuine Note"
        else:
            result = "⚠️ Suspicious Note"

        confidence = f"{genuine_prob:.2f} | PatternSim: {similarity:.2f}"

        os.remove(temp_path)
        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
