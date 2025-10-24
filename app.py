from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
os.makedirs(STATIC_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "smishing_svm.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "combined_dataset.csv")
CONFUSION_MATRIX_PATH = os.path.join(STATIC_FOLDER, "confusion_matrix.png")

# Globals
model, vectorizer, acc, report_dict = None, None, 0, {}
status_message = "App starting..."
SMISH_KEYWORDS = ["http", "https", "click", "login", "password", "account", "verify", "urgent", "free", "win"]

# --- Helper Functions ---
def preprocess_message(msg):
    msg = str(msg).lower()
    msg = re.sub(r"http\S+", "", msg)
    msg = re.sub(r"[^\w\s]", "", msg)
    return msg.strip()

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Smish"], yticklabels=["Legit", "Smish"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

def check_files_and_load_model():
    global model, vectorizer, acc, report_dict, status_message
    missing = [p for p in [MODEL_PATH, VECTORIZER_PATH, DATASET_PATH] if not os.path.exists(p)]
    if missing:
        status_message = f"⚠️ Missing: {', '.join(missing)}"
        return

    try:
        df = pd.read_csv(DATASET_PATH).dropna(subset=["message", "label"])
        df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        X_tfidf = vectorizer.transform(df["message"].apply(preprocess_message))
        y_pred = model.predict(X_tfidf)
        acc = accuracy_score(df["label_num"], y_pred)
        report = classification_report(df["label_num"], y_pred, target_names=["Legit", "Smish"], output_dict=True)
        report_dict = pd.DataFrame(report).T.to_dict("index")

        save_confusion_matrix(df["label_num"], y_pred)
        status_message = f"✅ Model ready (Accuracy: {acc*100:.2f}%)"
    except Exception as e:
        status_message = f"❌ Error loading model: {e}"

# Load model
check_files_and_load_model()

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html", acc=f"{acc*100:.2f}", report=report_dict, status=status_message)

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": status_message}), 500

    message = request.form.get("message") or (request.json.get("message") if request.is_json else None)
    if not message:
        return jsonify({"error": "No message provided"}), 400

    clean_msg = preprocess_message(message)
    X = vectorizer.transform([clean_msg])
    pred_score = model.decision_function(X)[0]
    keyword_count = sum(clean_msg.count(k) for k in SMISH_KEYWORDS)

    if keyword_count >= 1:
        prediction = 1
        reason = f"Keyword trigger ({keyword_count} found)"
    else:
        prediction = 1 if pred_score > 0 else 0
        reason = "Model decision"

    label = "🚨 Smishing / Spam" if prediction == 1 else "✅ Legitimate"

    return jsonify({
        "message": message,
        "prediction": label,
        "clean_message": clean_msg,
        "decision_score": float(pred_score),
        "keyword_count": keyword_count,
        "reason": reason
    })

@app.route("/visualize")
def visualize():
    return render_template("visualize.html", acc=f"{acc*100:.2f}")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
