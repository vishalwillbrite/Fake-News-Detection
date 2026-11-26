"""
Automated test suite for VERITAS.AI Continuous Learning & Dataset Management.
Uses temporary directories and mocked external APIs so production datasets
and models are never altered during testing.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_text
from services.label_validation_service import validate_candidate_label
from services.model_retraining_service import ModelRetrainingService
from services.news_collection_service import NewsCollectionService
from services.pipeline_storage import PipelineStorage
from services.scheduler_service import ContinuousLearningScheduler
from services.search_provider import SearchArticle


class TestContinuousLearningPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

        # Build small synthetic original Fake.csv / True.csv
        fake_rows = [
            {"title": f"Shocking alien hoax {i}", "text": f"Secret alien hoax conspiracy leaked shockingly fake story number {i}.", "subject": "News", "date": "2026-01-01"}
            for i in range(15)
        ]
        true_rows = [
            {"title": f"Official NASA climate report {i}", "text": f"NASA scientists published official climate satellite telemetry report number {i}.", "subject": "Science", "date": "2026-01-01"}
            for i in range(15)
        ]
        self.fake_csv = os.path.join(self.root, "Fake.csv")
        self.true_csv = os.path.join(self.root, "True.csv")
        pd.DataFrame(fake_rows).to_csv(self.fake_csv, index=False)
        pd.DataFrame(true_rows).to_csv(self.true_csv, index=False)

        # Train baseline model v1.0
        combined = pd.concat([
            pd.DataFrame(fake_rows).assign(label=0),
            pd.DataFrame(true_rows).assign(label=1),
        ], ignore_index=True)
        vec = TfidfVectorizer(max_features=5000)
        X = vec.fit_transform(combined["text"].apply(preprocess_text))
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, combined["label"])

        self.prod_model = os.path.join(self.root, "fake_news_model.pkl")
        self.prod_vec = os.path.join(self.root, "tfidf_vectorizer.pkl")
        joblib.dump(clf, self.prod_model)
        joblib.dump(vec, self.prod_vec)

        self.storage = PipelineStorage(
            base_dir=os.path.join(self.root, "data_pipeline"),
            fake_csv=self.fake_csv,
            true_csv=self.true_csv,
            prod_model_path=self.prod_model,
            prod_vectorizer_path=self.prod_vec,
            model_versions_dir=os.path.join(self.root, "versions"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_label_validation_never_assumes_gdelt_is_real(self):
        # Plain news without ClaimReview -> NEEDS_REVIEW (never auto-REAL)
        res_plain = validate_candidate_label(
            title="New satellite launched",
            content="NASA launched an Earth observation satellite on Tuesday morning.",
            url="https://example.com/news/1",
            raw_html="<html><body><p>NASA launched an Earth observation satellite.</p></body></html>",
        )
        self.assertIsNone(res_plain.proposed_label)
        self.assertEqual(res_plain.status, "NEEDS_REVIEW")

        # Structured ClaimReview False -> PENDING with label 0
        html_cr = """
        <html><head><script type="application/ld+json">
        {"@context":"https://schema.org","@type":"ClaimReview",
         "claimReviewed":"Aliens built pyramids yesterday",
         "reviewRating":{"@type":"Rating","alternateName":"False"},
         "author":{"@type":"Organization","name":"FactCheckOrg"}}
        </script></head><body><p>Fact check article body.</p></body></html>
        """
        res_cr = validate_candidate_label("Fact check", "Fact check article body.", "https://factcheck.org/1", html_cr)
        self.assertEqual(res_cr.proposed_label, 0)
        self.assertEqual(res_cr.status, "PENDING")

    def test_collection_deduplication_approval_retraining_and_rollback(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock_gdelt"
        mock_provider.search.return_value = [
            SearchArticle(
                title="NASA climate observatory update",
                url="https://example.org/nasa-1",
                domain="example.org",
                publisher="Example",
                published_date="2026-05-01",
            ),
            SearchArticle(
                title="Duplicate NASA climate observatory update",
                url="https://example.org/nasa-1",  # duplicate URL
                domain="example.org",
                publisher="Example",
                published_date="2026-05-01",
            ),
        ]

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.text = (
            "<html><body><article><p>NASA scientists released new orbital climate telemetry "
            "measurements confirming global ocean surface monitoring improvements in 2026.</p></article></body></html>"
        )
        mock_session.get.return_value = mock_resp

        collector = NewsCollectionService(storage=self.storage, providers=[mock_provider], session=mock_session)
        summary = collector.collect_news(queries=["NASA climate"], max_per_query=2, trigger_type="manual")
        self.assertTrue(summary["success"])
        self.assertEqual(summary["new_candidates_count"], 1)
        self.assertEqual(summary["duplicates_skipped"], 1)

        # Original Fake.csv and True.csv remain untouched (15 + 15)
        self.assertEqual(len(pd.read_csv(self.fake_csv)), 15)
        self.assertEqual(len(pd.read_csv(self.true_csv)), 15)

        # Candidate should be NEEDS_REVIEW
        needs_rev = self.storage.list_candidates(status="NEEDS_REVIEW")
        self.assertEqual(len(needs_rev), 1)
        cid = needs_rev[0]["id"]

        # Retraining below threshold must be skipped
        reloader_calls = []
        retrainer = ModelRetrainingService(
            storage=self.storage,
            min_approved_samples=2,
            on_model_updated=lambda m, v: reloader_calls.append((m, v)),
        )
        skip_res = retrainer.retrain_if_ready()
        self.assertTrue(skip_res.get("skipped"))

        # Manually approve the collected article as REAL (1) and add a 2nd approved FAKE (0)
        self.assertTrue(self.storage.update_candidate_status(cid, "APPROVED", label=1, reviewed_by="admin"))
        cid2 = self.storage.add_candidate(
            url="https://example.org/hoax-2",
            title="Hoax alien pyramid story",
            content="Secret alien hoax conspiracy leaked shockingly fake story about underwater pyramids.",
            publisher="Example",
            domain="example.org",
            published_date="2026-05-02",
            proposed_label=0,
            status="APPROVED",
            label_source="human_review",
            verification_evidence={"note": "Verified false"},
        )
        self.assertIsNotNone(cid2)

        # Trigger retraining now that 2 approved samples exist
        train_res = retrainer.retrain_if_ready()
        self.assertTrue(train_res["success"])
        self.assertTrue(train_res["deployed"])
        self.assertEqual(train_res["model_version"], "v1.1")
        self.assertEqual(len(reloader_calls), 1)

        # Test rollback back to v1.0
        rb = retrainer.rollback_to_version("v1.0")
        self.assertTrue(rb["success"])
        self.assertEqual(self.storage.get_active_model_version()["version"], "v1.0")
        self.assertEqual(len(reloader_calls), 2)

    def test_scheduler_start_and_stop(self):
        mock_collector = MagicMock()
        mock_retrainer = MagicMock()
        orig_env = os.environ.pop("VERITAS_DISABLE_SCHEDULER", None)
        try:
            sched = ContinuousLearningScheduler(mock_collector, mock_retrainer, interval_minutes=60)
            sched.start()
            self.assertTrue(sched.is_running())
            sched._run_scheduled_cycle()
            mock_collector.collect_news.assert_called_once_with(trigger_type="scheduled")
            mock_retrainer.retrain_if_ready.assert_called_once()
            sched.stop()
        finally:
            if orig_env is not None:
                os.environ["VERITAS_DISABLE_SCHEDULER"] = orig_env

    def test_admin_routes_protected_from_ordinary_users(self):
        os.environ["VERITAS_DISABLE_SCHEDULER"] = "1"
        import app as flask_app

        client = flask_app.app.test_client()
        # Unauthenticated API calls must return 401
        self.assertEqual(client.get("/api/admin/stats").status_code, 401)
        self.assertEqual(client.post("/api/admin/collect").status_code, 401)
        self.assertEqual(client.post("/api/admin/retrain").status_code, 401)

        # Authenticated with X-Admin-Key succeeds
        r_ok = client.get("/api/admin/stats", headers={"X-Admin-Key": flask_app.ADMIN_ACCESS_KEY})
        self.assertEqual(r_ok.status_code, 200)
        self.assertTrue(r_ok.get_json()["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
