from flask import Flask, render_template, request
import joblib
import time
from preprocessing import preprocess_text

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

    start_time = time.time()

    prediction = model.predict(news_vector)[0]

    probability = model.predict_proba(news_vector)

    confidence = round(max(probability[0]) * 100, 2)

    prediction_time = round(time.time() - start_time, 4)

    if prediction == 0:
        result = "❌ Fake News"
        result_class = "fake"
    else:
        result = "✅ Real News"
        result_class = "real"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        result_class=result_class,
        prediction_time=prediction_time
    )
    if not news.strip():
        return render_template(
        "dashboard.html",
        error="Please enter a news article."
    )
    


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)