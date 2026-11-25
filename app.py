from flask import Flask, render_template, request, jsonify
import joblib
import time
from preprocessing import preprocess_text

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def analyze_article(news_text):
    """
    Helper function to run preprocessing, vectorization, and inference.
    Returns a dictionary of analysis metrics.
    """
    start_time = time.time()
    
    cleaned_news = preprocess_text(news_text)
    news_vector = vectorizer.transform([cleaned_news])
    
    prediction = int(model.predict(news_vector)[0])
    probabilities = model.predict_proba(news_vector)[0]
    
    fake_prob = round(float(probabilities[0]) * 100, 2)
    real_prob = round(float(probabilities[1]) * 100, 2)
    confidence = round(max(probabilities) * 100, 2)
    latency = round(time.time() - start_time, 4)
    
    if prediction == 0:
        label = "Fake News"
        result_icon = "❌"
        result_class = "fake"
        verdict_summary = "High probability of deceptive patterns, unverified claims, or sensationalist rhetoric."
    else:
        label = "Real News"
        result_icon = "✅"
        result_class = "real"
        verdict_summary = "Linguistic markers closely align with verified journalistic standards and factual reporting."

    # Extract notable vocabulary tokens found in the TF-IDF feature space
    feature_names = vectorizer.get_feature_names_out()
    vector_array = news_vector.toarray()[0]
    top_indices = vector_array.argsort()[::-1][:10]
    keywords = [
        {"word": feature_names[i], "weight": round(float(vector_array[i]), 4)}
        for i in top_indices if vector_array[i] > 0
    ]

    word_count = len(news_text.split())
    char_count = len(news_text)
    cleaned_word_count = len(cleaned_news.split())

    return {
        "success": True,
        "prediction": prediction,
        "label": label,
        "result_icon": result_icon,
        "result_class": result_class,
        "verdict_summary": verdict_summary,
        "confidence": confidence,
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "latency": latency,
        "word_count": word_count,
        "char_count": char_count,
        "cleaned_word_count": cleaned_word_count,
        "keywords": keywords,
        "news_snippet": news_text[:350] + ("..." if len(news_text) > 350 else ""),
        "news_full": news_text
    }


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ---------------- PREDICT (WEB & FORM) ----------------
@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "").strip()

    if not news:
        return render_template(
            "dashboard.html",
            error="Please paste or enter a news article to analyze."
        )

    data = analyze_article(news)

    # Check if request prefers JSON (e.g., fetch from dynamic JS HUD)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
        return jsonify(data)

    return render_template(
        "result.html",
        prediction=f"{data['result_icon']} {data['label']}",
        label=data["label"],
        confidence=data["confidence"],
        result_class=data["result_class"],
        verdict_summary=data["verdict_summary"],
        real_prob=data["real_prob"],
        fake_prob=data["fake_prob"],
        prediction_time=data["latency"],
        word_count=data["word_count"],
        char_count=data["char_count"],
        cleaned_word_count=data["cleaned_word_count"],
        keywords=data["keywords"],
        news_snippet=data["news_snippet"],
        news_full=data["news_full"]
    )


# ---------------- API PREDICT (JSON) ----------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or request.form
    news = payload.get("news", "").strip()

    if not news:
        return jsonify({"success": False, "error": "No news text provided."}), 400

    data = analyze_article(news)
    return jsonify(data)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
