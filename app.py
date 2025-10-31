from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tempfile

app = Flask(__name__)

def preprocess_image(path, size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image format.")
    img = cv2.resize(img, size)
    return img

def extract_wavelet_features(img, wavelet_name='db2'):
    coeffs2 = pywt.dwt2(img, wavelet_name)
    cA, (cH, cV, cD) = coeffs2
    features = []
    for mat in [cH, cV, cD]:
        hist, _ = np.histogram(mat.flatten(), bins=64, density=True)
        features.extend([np.var(hist), skew(hist), kurtosis(hist)])
    return np.array(features)

dataset_path = "features_dataset.npz"
data = np.load(dataset_path)
X, y = data["X"], data["y"]
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

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

        img = preprocess_image(temp_path)
        features = extract_wavelet_features(img).reshape(1, -1)
        proba = lda.predict_proba(features)[0]
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
        return render_template("result.html", result=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
