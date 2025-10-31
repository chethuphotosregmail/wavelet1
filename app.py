from flask import Flask, render_template, request
import cv2, os, tempfile, numpy as np, pywt
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ---------- 1.  IMAGE PREPROCESS ----------
def preprocess_image(path, size=(1080,1080)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid or unreadable image.")
    # keep color for RGB features
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return img, gray


# ---------- 2.  FEATURE EXTRACTORS ----------
def wavelet_color_features(img, wavelet_name='db2'):
    """Wavelet stats on each RGB channel"""
    feats=[]
    for channel in cv2.split(img):
        coeffs2=pywt.dwt2(channel,wavelet_name)
        cA,(cH,cV,cD)=coeffs2
        for mat in [cH,cV,cD]:
            hist,_=np.histogram(mat.flatten(),bins=64,density=True)
            feats.extend([np.var(hist),skew(hist),kurtosis(hist)])
    return feats

def lbp_texture_features(gray):
    """Local Binary Pattern histogram (micro-texture)"""
    lbp = local_binary_pattern(gray,P=8,R=1,method="uniform")
    hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0,59), density=True)
    return hist.tolist()

def extract_features(img_color, gray):
    f1 = wavelet_color_features(img_color)
    f2 = lbp_texture_features(gray)
    return np.array(f1+f2)


# ---------- 3.  LOAD DATASET + TRAIN ----------
dataset_path="features_dataset.npz"
if not os.path.exists(dataset_path):
    raise FileNotFoundError("⚠️ features_dataset.npz missing.")

print("📂 Loading dataset...")
data=np.load(dataset_path,allow_pickle=True)
X,y=data["X"],data["y"]
X=np.nan_to_num(X)

# scale & split
scaler=StandardScaler()
X=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# stronger model
rf=RandomForestClassifier(n_estimators=250,max_depth=None,random_state=42)
rf.fit(X_train,y_train)
print("✅ RandomForest trained successfully!")


# ---------- 4.  FLASK ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    try:
        file=request.files.get("file")
        if not file or file.filename=="":
            return render_template("result.html",result="⚠️ No image selected!")

        # save temporarily
        temp_path=os.path.join(tempfile.gettempdir(),file.filename)
        file.save(temp_path)

        # preprocess + feature extract
        img_color,gray=preprocess_image(temp_path)
        feats=extract_features(img_color,gray).reshape(1,-1)
        feats=scaler.transform(feats)

        # predict
        proba=rf.predict_proba(feats)[0]
        genuine_prob=float(proba[1])

        if genuine_prob>=0.9:
            result="✅ Genuine Note"
        elif genuine_prob<=0.6:
            result="❌ Fake (Xerox) Note"
        else:
            result="⚠️ Suspicious Note"

        conf=f"{genuine_prob:.2f}"
        os.remove(temp_path)
        return render_template("result.html",result=result,confidence=conf)

    except Exception as e:
        print("Error:",e)
        return render_template("result.html",result=f"❌ Error: {str(e)}")


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001,debug=True)
