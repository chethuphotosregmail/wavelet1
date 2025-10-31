from flask import Flask, render_template, request
import cv2
import os
import tempfile
import numpy as np
import pywt
import joblib
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier  # for type clarity only

app = Flask(__name__)

# -------------------- PARAMETERS --------------------
ROI_SIZE = 1080                     # final crop size (square)
TEMPLATE_FOLDER = "ref_templates"   # optional: place small reference patches here
USE_TEMPLATE_FIRST = True           # try template matching first (if templates exist)
NUM_TOP_WINDOWS = 1                 # how many top texture windows to consider (1 is enough)
WINDOW_SIZE_RATIO = 0.25            # sliding window size relative to image (0.25 => window covers 25% of shorter side)
PADDING = 20                        # extra pixels padding around detected ROI when cropping

# -------------------- PREPROCESS & FEATURE HELPERS --------------------
def preprocess_and_crop(path):
    """
    Read image from path, detect ROI automatically (template-match or texture energy),
    crop to a square region around the detected ROI, return (img_color_crop, gray_crop)
    resized to ROI_SIZE x ROI_SIZE (and equalized grayscale).
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid or unreadable image.")

    # Make a copy for template matching / scanning
    img_rgb = img.copy()
    h, w = img.shape[:2]

    # Try template matching if templates exist and enabled
    if USE_TEMPLATE_FIRST and os.path.isdir(TEMPLATE_FOLDER) and len(os.listdir(TEMPLATE_FOLDER)) > 0:
        best = template_match_multi(img_rgb, TEMPLATE_FOLDER)
        if best is not None:
            x, y, bw, bh = best  # bounding box center/size returned by template_match_multi
            roi = smart_crop(img_rgb, x, y, bw, bh, padding=PADDING)
            return finalize_crop(roi)

    # Otherwise fallback to texture-energy scanning
    win = max(32, int(min(h, w) * WINDOW_SIZE_RATIO))
    # compute texture energy map (Laplacian variance)
    center_x, center_y, bw, bh = detect_roi_by_texture(img_rgb, win=win, top_k=NUM_TOP_WINDOWS)
    roi = smart_crop(img_rgb, center_x, center_y, bw, bh, padding=PADDING)
    return finalize_crop(roi)


def finalize_crop(roi_img):
    """
    Convert ROI to color + equalized grayscale and resize to ROI_SIZE.
    """
    # ensure ROI is large enough; if not, pad it
    rh, rw = roi_img.shape[:2]
    if rh == 0 or rw == 0:
        raise ValueError("Empty ROI detected.")

    # Make square: center crop/pad to make square before final resize
    side = max(rh, rw)
    square = np.zeros((side, side, 3), dtype=roi_img.dtype) + 255
    y0 = (side - rh) // 2
    x0 = (side - rw) // 2
    square[y0:y0+rh, x0:x0+rw] = roi_img

    # Resize to ROI_SIZE
    final_color = cv2.resize(square, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
    final_gray = cv2.cvtColor(final_color, cv2.COLOR_BGR2GRAY)
    final_gray = cv2.equalizeHist(final_gray)
    return final_color, final_gray


# -------------------- TEMPLATE MATCHING --------------------
def template_match_multi(img, template_folder, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """
    Try template matching using all images in template_folder across multiple scales.
    Returns best match as (center_x, center_y, w, h) or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_box = None

    for fname in os.listdir(template_folder):
        tpath = os.path.join(template_folder, fname)
        tpl = cv2.imread(tpath)
        if tpl is None:
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        th, tw = tpl_gray.shape[:2]

        # Try multiple scales of the template
        for scale in scales:
            new_w = max(8, int(tw * scale))
            new_h = max(8, int(th * scale))
            tpl_resized = cv2.resize(tpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if tpl_resized.shape[0] >= gray.shape[0] or tpl_resized.shape[1] >= gray.shape[1]:
                continue

            res = cv2.matchTemplate(gray, tpl_resized, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

            if maxVal > best_score and maxVal > 0.5:
                best_score = maxVal
                top_left = maxLoc
                center_x = top_left[0] + new_w // 2
                center_y = top_left[1] + new_h // 2
                best_box = (center_x, center_y, new_w, new_h)

    return best_box


# -------------------- TEXTURE-ENERGY (LAPLACIAN VARIANCE) ROI DETECTION --------------------
def detect_roi_by_texture(img, win=200, top_k=1):
    """
    Slide a window across the image, compute Laplacian variance per window,
    return center coordinates of the top-scoring window, plus box width & height.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    step_x = max(16, win // 4)
    step_y = max(16, win // 4)

    scores = []
    boxes = []

    # ensure we cover entire image
    for y in range(0, h - win + 1, step_y):
        for x in range(0, w - win + 1, step_x):
            patch = gray[y:y+win, x:x+win]
            # compute Laplacian variance (local high-frequency energy)
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            score = lap.var()
            scores.append(score)
            boxes.append((x, y, win, win))

    if len(scores) == 0:
        # fallback entire image
        return w//2, h//2, w, h

    # pick top K clusters by score, compute centroid of their boxes
    idxs = np.argsort(scores)[-top_k:]
    xs = []
    ys = []
    bw = win
    bh = win
    for i in idxs:
        x, y, _, _ = boxes[i]
        xs.append(x + win // 2)
        ys.append(y + win // 2)

    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))
    return center_x, center_y, bw, bh


# -------------------- SMART CROP --------------------
def smart_crop(img, cx, cy, bw, bh, padding=20):
    """
    Crop a region centered at (cx, cy) with box size bw x bh, add padding and
    keep bounds within image.
    """
    h, w = img.shape[:2]
    half_w = bw // 2
    half_h = bh // 2
    x1 = max(0, int(cx - half_w - padding))
    y1 = max(0, int(cy - half_h - padding))
    x2 = min(w, int(cx + half_w + padding))
    y2 = min(h, int(cy + half_h + padding))
    return img[y1:y2, x1:x2].copy()


# -------------------- FEATURE EXTRACTION (same as your latest) --------------------
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


# -------------------- LOAD MODEL & SCALER --------------------
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    raise FileNotFoundError("⚠️ Run train_model.py first to generate model.pkl and scaler.pkl")

rf = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


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

        # detect & crop automatically
        img_color_crop, gray_crop = preprocess_and_crop(temp_path)

        # extract, scale, predict
        feats = extract_features(img_color_crop, gray_crop).reshape(1, -1)
        feats = scaler.transform(feats)
        proba = rf.predict_proba(feats)[0]
        genuine_prob = float(proba[1])

        if genuine_prob >= 0.9:
            result = "✅ Genuine"
        elif genuine_prob <= 0.6:
            result = "❌ Fake"
        else:
            result = "⚠️ Suspicious"

        confidence = f"{genuine_prob:.2f}"

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        print("Error:", e)
        return render_template("result.html", result=f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
