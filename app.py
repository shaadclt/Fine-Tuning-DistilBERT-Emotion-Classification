from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import os

app = Flask(__name__)

# Load model directly from Hugging Face Hub
classifier = pipeline(
    "text-classification",
    model="shaadclt/distilbert-emotion-classifier"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    result = classifier(text)[0]
    return jsonify({
        "emotion": result["label"],
        "confidence": round(result["score"] * 100, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
