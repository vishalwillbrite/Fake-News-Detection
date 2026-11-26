"""
Automated test suite for VERITAS.AI External News Verification Upgrade.
Covers claim extraction, GDELT search, webpage retrieval, NLI evidence verification,
final assessment logic, Flask routes, and existing ML regression.
"""

import time
import unittest
from unittest.mock import MagicMock

import app as flask_app
from services.claim_extraction_service import extract_claims
from services.evidence_retrieval_service import EvidenceRetrievalService, RetrievedEvidence
from services.evidence_verification_service import EvidenceVerificationService
from services.final_decision_service import compute_final_assessment
from services.gdelt_service import GDELTSearchProvider
from services.search_provider import SearchArticle
from services.verification_orchestrator import VerificationOrchestrator


class TestClaimExtraction(unittest.TestCase):
    def test_extracts_factual_claims_and_queries(self):
        article = (
            "NASA launched the Artemis weather satellite from Cape Canaveral in 2026. "
            "I think this is super exciting! "
            "Click here to subscribe now. "
            "The European Space Agency confirmed a $420 million budget increase for climate monitoring."
        )
        claims = extract_claims(article, max_claims=4)
        self.assertGreaterEqual(len(claims), 2)
        texts = " ".join(c.text for c in claims)
        self.assertIn("NASA launched", texts)
        self.assertIn("European Space Agency", texts)
        self.assertNotIn("subscribe now", texts.lower())
        for c in claims:
            self.assertTrue(len(c.search_query.split()) >= 2)


class TestGDELTSearchProvider(unittest.TestCase):
    def test_successful_search_deduplication_and_urls(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"articles": []}'
        mock_resp.json.return_value = {
            "articles": [
                {
                    "url": "https://www.reuters.com/science/nasa-weather-satellite-2026-05-01/",
                    "title": "NASA launches next-generation weather satellite into orbit",
                    "seendate": "20260501T100000Z",
                    "domain": "www.reuters.com",
                    "language": "English",
                },
                {
                    "url": "https://www.reuters.com/science/nasa-weather-satellite-2026-05-01/",
                    "title": "NASA launches next-generation weather satellite into orbit",
                    "seendate": "20260501T100000Z",
                    "domain": "www.reuters.com",
                    "language": "English",
                },
                {
                    "url": "https://www.bbc.com/news/unrelated-cooking-recipe",
                    "title": "Best chocolate cake recipes for weekend baking",
                    "seendate": "20260501T090000Z",
                    "domain": "www.bbc.com",
                    "language": "English",
                },
            ]
        }
        mock_session.get.return_value = mock_resp

        provider = GDELTSearchProvider(min_request_interval=0.0, session=mock_session)
        results = provider.search("NASA weather satellite orbit", "NASA launched a weather satellite into orbit.")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://www.reuters.com/science/nasa-weather-satellite-2026-05-01/")
        self.assertEqual(results[0].publisher, "REUTERS" if False else "Reuters")
        self.assertEqual(results[0].published_date, "2026-05-01")

    def test_empty_and_non_json_responses(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Please limit requests to one every 5 seconds."
        mock_session.get.return_value = mock_resp

        provider = GDELTSearchProvider(min_request_interval=0.0, session=mock_session)
        self.assertEqual(provider.search("NASA satellite", max_results=3), [])


class TestEvidenceRetrievalService(unittest.TestCase):
    def test_successful_retrieval_and_inaccessible_websites(self):
        mock_session = MagicMock()

        def fake_get(url, **kwargs):
            resp = MagicMock()
            if "goodnews.org" in url:
                resp.status_code = 200
                resp.encoding = "utf-8"
                resp.headers = {"Content-Type": "text/html; charset=utf-8"}
                html_body = (
                    "<html><head><title>Good News</title></head><body>"
                    "<script>alert('xss')</script>"
                    "<article><p>NASA successfully launched the Artemis weather satellite into orbit "
                    "on Monday morning to improve global storm forecasting.</p></article>"
                    "</body></html>"
                ).encode("utf-8")
                resp.iter_content.return_value = [html_body]
                return resp
            elif "blockednews.org" in url:
                resp.status_code = 403
                resp.headers = {"Content-Type": "text/html"}
                return resp
            raise TimeoutError("Simulated timeout")

        mock_session.get.side_effect = fake_get

        retriever = EvidenceRetrievalService(respect_robots=False, session=mock_session)
        articles = [
            SearchArticle(
                title="NASA launches weather satellite",
                url="https://goodnews.org/nasa-launch",
                domain="goodnews.org",
                publisher="Goodnews",
                published_date="2026-05-01",
            ),
            SearchArticle(
                title="Paywalled NASA article",
                url="https://blockednews.org/nasa-paywall",
                domain="blockednews.org",
                publisher="Blockednews",
            ),
        ]
        evidence = retriever.retrieve_evidence_for_claim("NASA launched the Artemis weather satellite.", articles)
        self.assertEqual(len(evidence), 2)
        self.assertTrue(evidence[0].accessible)
        self.assertIn("NASA successfully launched", evidence[0].evidence_passage)
        self.assertNotIn("<script>", evidence[0].evidence_passage)
        self.assertFalse(evidence[1].accessible)
        self.assertEqual(evidence[1].evidence_passage, "")
        self.assertIn("403", evidence[1].status_note)

        # Inaccessible source must go to unretrieved_sources and NOT be scored by NLI
        verifier = EvidenceVerificationService(
            nli_scorer=lambda p, h: {"entailment": 0.90, "contradiction": 0.04, "neutral": 0.06}
        )
        c_res = verifier.verify_claim(1, "NASA launched the Artemis weather satellite.", "NASA Artemis", evidence)
        self.assertEqual(len(c_res.supporting_evidence), 1)
        self.assertEqual(len(c_res.unretrieved_sources), 1)
        self.assertEqual(c_res.unretrieved_sources[0].stance, "UNRETRIEVED")


class TestEvidenceVerificationAndFinalDecision(unittest.TestCase):
    def _ev(self, url, passage, accessible=True):
        return RetrievedEvidence(
            title="Source Title",
            url=url,
            domain="example.com",
            publisher="Example",
            published_date="2026-05-01",
            evidence_passage=passage,
            accessible=accessible,
        )

    def test_supported_contradicted_mixed_and_insufficient(self):
        def fake_nli(premise, hypothesis):
            p = premise.lower()
            if "confirmed" in p:
                return {"entailment": 0.91, "contradiction": 0.03, "neutral": 0.06}
            if "false" in p or "never happened" in p:
                return {"entailment": 0.04, "contradiction": 0.89, "neutral": 0.07}
            return {"entailment": 0.15, "contradiction": 0.10, "neutral": 0.75}

        verifier = EvidenceVerificationService(nli_scorer=fake_nli)

        # 1. SUPPORTED
        c_sup = verifier.verify_claim(1, "City opened a new rail line.", "City rail line", [
            self._ev("https://a.com/1", "Officials confirmed the city opened a new rail line.")
        ])
        self.assertEqual(c_sup.label, "SUPPORTED")

        # 2. CONTRADICTED
        c_con = verifier.verify_claim(2, "City closed all schools.", "City schools closed", [
            self._ev("https://a.com/2", "The rumor is false; schools never happened to close.")
        ])
        self.assertEqual(c_con.label, "CONTRADICTED")

        # 3. MIXED (conflicting evidence preserved)
        c_mix = verifier.verify_claim(3, "Tax rate dropped to 5%.", "Tax rate 5%", [
            self._ev("https://a.com/3", "Ministry confirmed tax rate dropped to 5%."),
            self._ev("https://b.com/4", "Audit says the 5% claim is false and rates rose."),
        ])
        self.assertEqual(c_mix.label, "MIXED")
        self.assertEqual(len(c_mix.supporting_evidence), 1)
        self.assertEqual(len(c_mix.contradicting_evidence), 1)

        # 4. INSUFFICIENT EVIDENCE (missing search results + neutral snippet)
        c_ins_empty = verifier.verify_claim(4, "Claim with no hits.", "Claim hits", [])
        self.assertEqual(c_ins_empty.label, "INSUFFICIENT EVIDENCE")

        c_ins_neu = verifier.verify_claim(5, "Claim with neutral hit.", "Claim neutral", [
            self._ev("https://a.com/5", "The weather today was sunny across the region.")
        ])
        self.assertEqual(c_ins_neu.label, "INSUFFICIENT EVIDENCE")

        # Final Decision Engine checks (including ML vs External disagreement!)
        f_real = compute_final_assessment(ml_prediction=1, ml_confidence=95.0, claim_results=[c_sup])
        self.assertEqual(f_real.final_label, "REAL")
        self.assertFalse(f_real.ml_external_disagreement)

        f_fake = compute_final_assessment(ml_prediction=0, ml_confidence=92.0, claim_results=[c_con])
        self.assertEqual(f_fake.final_label, "FAKE")
        self.assertFalse(f_fake.ml_external_disagreement)

        # Disagreement: ML says Fake (0), External says Supported -> MIXED / NEEDS REVIEW
        f_disagree = compute_final_assessment(ml_prediction=0, ml_confidence=88.0, claim_results=[c_sup])
        self.assertEqual(f_disagree.final_label, "MIXED / NEEDS REVIEW")
        self.assertTrue(f_disagree.ml_external_disagreement)

        # Missing evidence must yield UNVERIFIED even if ML is 99% confident
        f_unver = compute_final_assessment(ml_prediction=0, ml_confidence=99.0, claim_results=[c_ins_empty])
        self.assertEqual(f_unver.final_label, "UNVERIFIED")


class TestFlaskAppRegressionAndOrchestrator(unittest.TestCase):
    def test_existing_routes_and_ml_prediction_preserved(self):
        client = flask_app.app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/dashboard").status_code, 200)

        r_pred = client.post("/predict", data={"news": "NASA launched a weather satellite into orbit."})
        self.assertEqual(r_pred.status_code, 200)

        r_api = client.post("/api/predict", json={"news": "NASA launched a weather satellite into orbit."})
        self.assertEqual(r_api.status_code, 200)
        data = r_api.get_json()
        self.assertTrue(data["success"])
        self.assertIn("confidence", data)
        self.assertIn("real_prob", data)
        self.assertIn("fake_prob", data)

    def test_background_verification_job_flow(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock_gdelt"
        mock_provider.search.return_value = [
            SearchArticle(
                title="NASA launches weather satellite",
                url="https://www.reuters.com/science/nasa-launch",
                domain="www.reuters.com",
                publisher="Reuters",
                published_date="2026-05-01",
            )
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve_evidence_for_claim.return_value = [
            RetrievedEvidence(
                title="NASA launches weather satellite",
                url="https://www.reuters.com/science/nasa-launch",
                domain="www.reuters.com",
                publisher="Reuters",
                published_date="2026-05-01",
                evidence_passage="NASA confirmed it launched a new weather satellite into orbit.",
                accessible=True,
            )
        ]

        mock_verifier = EvidenceVerificationService(
            nli_scorer=lambda p, h: {"entailment": 0.90, "contradiction": 0.04, "neutral": 0.06}
        )

        orchestrator = VerificationOrchestrator(
            providers=[mock_provider],
            retrieval_service=mock_retriever,
            verification_service=mock_verifier,
        )

        orig_orch = flask_app.verification_orchestrator
        flask_app.verification_orchestrator = orchestrator
        try:
            client = flask_app.app.test_client()
            resp = client.post("/api/verify", json={"news": "NASA launched a new weather satellite into orbit in 2026."})
            self.assertEqual(resp.status_code, 202)
            body = resp.get_json()
            job_id = body["job_id"]

            final_job = None
            for _ in range(30):
                st = client.get(f"/api/verify/status/{job_id}").get_json()
                if st["status"] == "completed":
                    final_job = st
                    break
                time.sleep(0.05)

            self.assertIsNotNone(final_job)
            result = final_job["result"]
            self.assertTrue(result["success"])
            self.assertEqual(result["sources"][0]["url"], "https://www.reuters.com/science/nasa-launch")
            self.assertEqual(result["claims"][0]["label"], "SUPPORTED")
        finally:
            flask_app.verification_orchestrator = orig_orch


if __name__ == "__main__":
    unittest.main(verbosity=2)
