from flask import Flask, render_template, request
import joblib
from preprocessing import preprocess_text

# Create Flask app
app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    cleaned_news = preprocess_text(news)

    news_vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_vector)[0]

    confidence = model.predict_proba(news_vector).max() * 100

    if prediction == 0:
        result = "❌ Fake News"
    else:
        result = "✅ Real News"

    return render_template(
        "result.html",
        prediction=result,
        confidence=f"{confidence:.2f}"
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)