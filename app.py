from flask import Flask, render_template, request

import joblib

app = Flask(__name__)
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template("index.html", prediction=result)

    if __name__ == "__main__":
        app.run(debug=True)
