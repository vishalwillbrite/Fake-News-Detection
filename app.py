from functools import wraps
import hmac
import joblib
import logging
import os
import threading
import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from preprocessing import preprocess_text
from services.verification_orchestrator import VerificationOrchestrator
from services.pipeline_storage import PipelineStorage
from services.news_collection_service import NewsCollectionService
from services.model_retraining_service import ModelRetrainingService
from services.scheduler_service import ContinuousLearningScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = Flask(__name__)
app.secret_key = os.environ.get("VERITAS_SECRET_KEY", "veritas-dev-secret-change-in-prod")
ADMIN_ACCESS_KEY = os.environ.get("VERITAS_ADMIN_KEY", "veritas-admin-2026")
COLLECTION_INTERVAL_MINUTES = int(os.environ.get("VERITAS_COLLECTION_INTERVAL_MIN", "60"))
MIN_APPROVED_RETRAIN = int(os.environ.get("VERITAS_MIN_APPROVED_RETRAIN", "10"))

# Load trained model and vectorizer under lock so updates stay paired
_model_lock = threading.RLock()
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def reload_production_model(model_path: str, vectorizer_path: str):
    """Atomically reload matching LogisticRegression model and TF-IDF vectorizer."""
    global model, vectorizer
    new_model = joblib.load(model_path)
    new_vec = joblib.load(vectorizer_path)
    with _model_lock:
        model = new_model
        vectorizer = new_vec


# Initialize modular external verification orchestrator (lazy-loads NLI model on demand)
verification_orchestrator = VerificationOrchestrator()

# Initialize continuous learning pipeline services
pipeline_storage = PipelineStorage()
collection_service = NewsCollectionService(storage=pipeline_storage)
retraining_service = ModelRetrainingService(
    storage=pipeline_storage,
    min_approved_samples=MIN_APPROVED_RETRAIN,
    on_model_updated=reload_production_model,
)
learning_scheduler = ContinuousLearningScheduler(
    collection_service=collection_service,
    retraining_service=retraining_service,
    interval_minutes=COLLECTION_INTERVAL_MINUTES,
)

if os.environ.get("VERITAS_DISABLE_SCHEDULER") != "1" and (
    not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
):
    learning_scheduler.start()


def is_admin_authenticated() -> bool:
    if session.get("is_admin") is True:
        return True
    header_key = request.headers.get("X-Admin-Key") or ""
    if header_key and hmac.compare_digest(header_key, ADMIN_ACCESS_KEY):
        return True
    return False


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_admin_authenticated():
            return jsonify({"success": False, "error": "Admin authentication required."}), 401
        return fn(*args, **kwargs)
    return wrapper


def analyze_article(news_text):
    """
    Helper function to run preprocessing, vectorization, and inference.
    Returns a dictionary of analysis metrics.
    """
    start_time = time.time()
    
    cleaned_news = preprocess_text(news_text)
    with _model_lock:
        news_vector = vectorizer.transform([cleaned_news])
        prediction = int(model.predict(news_vector)[0])
        probabilities = model.predict_proba(news_vector)[0]
        feature_names = vectorizer.get_feature_names_out()
    
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
    news = (payload.get("news") or "").strip()

    if not news:
        return jsonify({"success": False, "error": "No news text provided."}), 400

    data = analyze_article(news)
    return jsonify(data)


# ---------------- API EXTERNAL VERIFY (BACKGROUND JOB) ----------------
@app.route("/api/verify", methods=["POST"])
def api_verify():
    payload = request.get_json(silent=True) or request.form
    news = (payload.get("news") or "").strip()

    if not news:
        return jsonify({"success": False, "error": "No news text provided."}), 400

    ml_result = analyze_article(news)
    job_id = verification_orchestrator.start_background_job(news, ml_result)
    return jsonify({
        "success": True,
        "job_id": job_id,
        "ml_analysis": ml_result,
    }), 202


@app.route("/api/verify/status/<job_id>", methods=["GET"])
def api_verify_status(job_id):
    job = verification_orchestrator.get_job_status(job_id)
    if job is None:
        return jsonify({"success": False, "error": "Verification job not found."}), 404
    return jsonify({"success": True, **job})


# ---------------- DATASET MANAGEMENT & CONTINUOUS LEARNING ADMIN ----------------
@app.route("/admin/dataset", methods=["GET", "POST"])
def admin_dataset():
    auth_error = None
    if request.method == "POST":
        action = request.form.get("action", "login")
        if action == "logout":
            session.pop("is_admin", None)
            return redirect(url_for("admin_dataset"))
        provided_key = request.form.get("admin_key", "")
        if provided_key and hmac.compare_digest(provided_key, ADMIN_ACCESS_KEY):
            session["is_admin"] = True
            return redirect(url_for("admin_dataset"))
        auth_error = "Invalid administrative access key."

    authenticated = is_admin_authenticated()
    stats = pipeline_storage.get_dashboard_stats() if authenticated else None
    return render_template(
        "admin_dataset.html",
        authenticated=authenticated,
        auth_error=auth_error,
        stats=stats,
        min_approved_retrain=retraining_service.min_approved_samples,
        scheduler_running=learning_scheduler.is_running(),
    )


@app.route("/api/admin/stats", methods=["GET"])
@admin_required
def api_admin_stats():
    stats = pipeline_storage.get_dashboard_stats()
    stats["min_approved_retrain"] = retraining_service.min_approved_samples
    stats["scheduler_running"] = learning_scheduler.is_running()
    return jsonify({"success": True, **stats})


@app.route("/api/admin/candidates", methods=["GET"])
@admin_required
def api_admin_candidates():
    status = request.args.get("status")
    limit = min(200, max(1, int(request.args.get("limit", 100))))
    items = pipeline_storage.list_candidates(status=status, limit=limit)
    return jsonify({"success": True, "candidates": items})


@app.route("/api/admin/collect", methods=["POST"])
@admin_required
def api_admin_collect():
    payload = request.get_json(silent=True) or {}
    queries = payload.get("queries")
    result = collection_service.collect_news(queries=queries, trigger_type="manual")
    code = 200 if result.get("success") else 409
    return jsonify(result), code


@app.route("/api/admin/candidates/<int:candidate_id>/review", methods=["POST"])
@admin_required
def api_admin_review_candidate(candidate_id):
    payload = request.get_json(silent=True) or request.form
    status = (payload.get("status") or "").strip().upper()
    raw_label = payload.get("label")
    label = int(raw_label) if raw_label in (0, 1, "0", "1") else None
    note = (payload.get("note") or "").strip()

    try:
        updated = pipeline_storage.update_candidate_status(
            candidate_id=candidate_id,
            status=status,
            label=label,
            reviewed_by="admin",
            review_note=note,
        )
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    if not updated:
        return jsonify({"success": False, "error": "Candidate article not found."}), 404
    return jsonify({"success": True, "candidate_id": candidate_id, "status": status, "label": label})


@app.route("/api/admin/retrain", methods=["POST"])
@admin_required
def api_admin_retrain():
    payload = request.get_json(silent=True) or {}
    min_samples = payload.get("min_samples")
    override = int(min_samples) if min_samples is not None else None
    result = retraining_service.retrain_if_ready(min_samples_override=override)
    return jsonify(result)


@app.route("/api/admin/rollback", methods=["POST"])
@admin_required
def api_admin_rollback():
    payload = request.get_json(silent=True) or {}
    version = (payload.get("version") or "").strip()
    if not version:
        return jsonify({"success": False, "error": "Target model version is required."}), 400
    result = retraining_service.rollback_to_version(version)
    code = 200 if result.get("success") else 400
    return jsonify(result), code


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
