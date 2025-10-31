from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tempfile

app = Flask(__name__)

# ---- 1. Preprocessing ----
def preprocess_image(path, size=(1080, 1080)):
    """
    Reads an image, converts it to grayscale, equalizes lighting,
    resizes to 1080x1080, and returns processed image.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid image format or unreadable file.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Equalize lighting for consistency (very important for camera images)
    gray = cv2.equalizeHist(gray)

    # Resize to 1080x1080
    resized = cv2.resize(gray, size)

    return resized


# ---- 2. Feature Extraction ----
def extract_wavelet_features(img, wavelet_name="db2"):
    """
    Extract variance, skewness, and kurtosis from DWT coefficients.
    """
    coeffs2 = pywt.dwt2(img, wavelet_name)
    cA, (cH, cV, cD) = coeffs2
    features = []
    for mat in [cH, cV, cD]:
        hist, _ = np.histogram(mat.flatten(), bins=64, density=True)
        features.extend([np.var(hist), skew(hist), kurtosis(hist)])
    return np.array(features)


# ---- 3. Load dataset & train model ----
dataset_path = "features_dataset.npz"
if not os.path.exists(dataset_path):
    raise FileNotFoundError("⚠️ features_dataset.npz not found! Please generate it first.")

print("📂 Loading dataset...")
data = np.load(dataset_path)
X, y = data["X"], data["y"]

# Clean any NaN/inf values
X = np.nan_to_num(X)

# Split for training/testing to avoid overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Normalize feature scale (very important for generalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print("✅ Model trained successfully!")


# ---- 4. Flask Routes ----
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

        # ---- Preprocess (Gray + Equalize + Resize) ----
        img = preprocess_image(temp_path)

        # ---- Feature Extraction ----
        features = extract_wavelet_features(img).reshape(1, -1)

        # ---- Scale features ----
        features = scaler.transform(features)

        # ---- Prediction ----
        proba = lda.predict_proba(features)[0]
        genuine_prob = float(proba[1])

        # ---- Decision Logic ----
        if genuine_prob >= 0.9:
            result = "✅ Genuine Note"
        elif genuine_prob <= 0.6:
            result = "❌ Fake Note"
        else:
            result = "⚠️ Suspicious Note"

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


# ---- 5. Run the Flask App ----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
