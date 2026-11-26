"""
model_retraining_service.py
Controlled ML retraining, holdout evaluation, deployment gating, versioning,
backup, and rollback for VERITAS.AI Logistic Regression + TF-IDF pipeline.
"""

import logging
import os
import shutil
import threading
from typing import Callable, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_text
from services.pipeline_storage import PipelineStorage, hash_content

logger = logging.getLogger(__name__)


class ModelRetrainingService:
    """
    Manages controlled retraining using the existing LogisticRegression +
    TfidfVectorizer + PorterStemmer pipeline, evaluated against a fixed holdout set.
    """

    def __init__(
        self,
        storage: PipelineStorage,
        min_approved_samples: int = 10,
        on_model_updated: Optional[Callable[[str, str], None]] = None,
    ):
        self.storage = storage
        self.min_approved_samples = min_approved_samples
        self.on_model_updated = on_model_updated
        self._lock = threading.Lock()

    def retrain_if_ready(self, min_samples_override: Optional[int] = None) -> dict:
        """
        Trigger retraining if at least `min_approved_samples` new APPROVED examples exist.
        Returns detailed comparison metrics and deployment decision.
        """
        threshold = min_samples_override if min_samples_override is not None else self.min_approved_samples

        if not self._lock.acquire(blocking=False):
            return {"success": False, "error": "Retraining is already in progress."}

        try:
            untrained_count = self.storage.count_untrained_approved()
            if untrained_count < threshold:
                return {
                    "success": False,
                    "skipped": True,
                    "reason": (
                        f"Insufficient new approved examples ({untrained_count}/{threshold} required). "
                        "Retraining skipped."
                    ),
                    "untrained_approved": untrained_count,
                    "min_required": threshold,
                }

            if not os.path.exists(self.storage.train_base_csv) or not os.path.exists(self.storage.holdout_csv):
                return {"success": False, "error": "Base training or holdout test dataset is missing."}

            # 1. Load base training split (strictly disjoint from holdout_test.csv)
            base_df = pd.read_csv(self.storage.train_base_csv)
            base_df["text"] = base_df["text"].fillna("").astype(str)
            base_df["label"] = base_df["label"].astype(int)
            if "content_hash" not in base_df.columns:
                base_df["content_hash"] = base_df["text"].apply(hash_content)

            # 2. Load only APPROVED candidates (never PENDING, REJECTED, or NEEDS_REVIEW)
            approved_rows = self.storage.get_approved_articles()
            approved_df = pd.DataFrame(
                [
                    {
                        "title": r["title"],
                        "text": r["content"],
                        "label": int(r["proposed_label"]),
                        "content_hash": r["content_hash"],
                        "candidate_id": r["id"],
                    }
                    for r in approved_rows
                ]
            )

            if approved_df.empty:
                return {"success": False, "skipped": True, "reason": "No valid approved articles available."}

            # Deduplicate and verify zero holdout leakage
            approved_df = approved_df[
                ~approved_df["content_hash"].apply(self.storage.is_in_holdout)
            ].drop_duplicates(subset=["content_hash"])

            combined_train = pd.concat(
                [base_df[["title", "text", "label", "content_hash"]], approved_df[["title", "text", "label", "content_hash"]]],
                ignore_index=True,
            ).drop_duplicates(subset=["content_hash"]).sample(frac=1, random_state=42).reset_index(drop=True)

            # 3. Record new dataset version snapshot
            ds_version = self.storage.next_dataset_version()
            snapshot_path = os.path.join(self.storage.datasets_dir, f"training_{ds_version}.csv")
            combined_train.to_csv(snapshot_path, index=False)

            real_cnt = int((combined_train["label"] == 1).sum())
            fake_cnt = int((combined_train["label"] == 0).sum())
            self.storage.record_dataset_version(
                version=ds_version,
                original_count=len(base_df),
                approved_added_count=len(approved_df),
                total_count=len(combined_train),
                real_count=real_cnt,
                fake_count=fake_cnt,
                snapshot_path=snapshot_path,
                notes=f"Added {len(approved_df)} approved verified articles",
            )

            used_ids = [int(cid) for cid in approved_df["candidate_id"].tolist()]
            self.storage.mark_approved_used_in_version(used_ids, ds_version)

            # 4. Fit TF-IDF Vectorizer + Logistic Regression together (exact existing algorithm)
            cleaned_texts = combined_train["text"].apply(preprocess_text)
            y_train = combined_train["label"].astype(int)

            candidate_vec = TfidfVectorizer(max_features=5000)
            X_train = candidate_vec.fit_transform(cleaned_texts)

            candidate_model = LogisticRegression(max_iter=1000)
            candidate_model.fit(X_train, y_train)

            # 5. Save candidate artifacts in versioned directory
            model_version = self.storage.next_model_version()
            ver_dir = os.path.join(self.storage.model_versions_dir, model_version)
            os.makedirs(ver_dir, exist_ok=True)
            cand_model_path = os.path.join(ver_dir, "fake_news_model.pkl")
            cand_vec_path = os.path.join(ver_dir, "tfidf_vectorizer.pkl")

            joblib.dump(candidate_model, cand_model_path)
            joblib.dump(candidate_vec, cand_vec_path)

            # 6. Evaluate both current production model & candidate model on fixed holdout_test.csv
            cand_metrics = self.storage.evaluate_artifacts_on_holdout(cand_model_path, cand_vec_path)
            active_info = self.storage.get_active_model_version()
            if active_info:
                curr_metrics = {
                    "accuracy": active_info["accuracy"],
                    "precision": active_info["precision_score"],
                    "recall": active_info["recall_score"],
                    "f1": active_info["f1_score"],
                    "confusion_matrix": active_info["confusion_matrix"],
                }
            else:
                curr_metrics = self.storage.evaluate_artifacts_on_holdout(
                    self.storage.prod_model_path,
                    self.storage.prod_vectorizer_path,
                )

            # 7. Deployment gate: deploy only if candidate meets/exceeds current model
            meets_criteria = (
                cand_metrics["f1"] >= curr_metrics["f1"]
                and cand_metrics["accuracy"] >= curr_metrics["accuracy"]
            )

            if meets_criteria:
                # Backup existing production artifacts before overwriting
                backup_dir = os.path.join(self.storage.model_versions_dir, "backup_latest")
                os.makedirs(backup_dir, exist_ok=True)
                if os.path.exists(self.storage.prod_model_path):
                    shutil.copy2(self.storage.prod_model_path, os.path.join(backup_dir, "fake_news_model.pkl"))
                if os.path.exists(self.storage.prod_vectorizer_path):
                    shutil.copy2(self.storage.prod_vectorizer_path, os.path.join(backup_dir, "tfidf_vectorizer.pkl"))

                shutil.copy2(cand_model_path, self.storage.prod_model_path)
                shutil.copy2(cand_vec_path, self.storage.prod_vectorizer_path)

                self.storage.record_model_version(
                    version=model_version,
                    dataset_version=ds_version,
                    model_path=cand_model_path,
                    vectorizer_path=cand_vec_path,
                    metrics=cand_metrics,
                    is_active=True,
                    status="DEPLOYED",
                    notes=f"Promoted: F1 {curr_metrics['f1']} -> {cand_metrics['f1']}",
                )
                if self.on_model_updated:
                    self.on_model_updated(self.storage.prod_model_path, self.storage.prod_vectorizer_path)
                deployed = True
                decision_msg = f"Candidate model {model_version} deployed (F1={cand_metrics['f1']}, Acc={cand_metrics['accuracy']})."
            else:
                self.storage.record_model_version(
                    version=model_version,
                    dataset_version=ds_version,
                    model_path=cand_model_path,
                    vectorizer_path=cand_vec_path,
                    metrics=cand_metrics,
                    is_active=False,
                    status="REJECTED_METRIC",
                    notes=(
                        f"Retained existing model: candidate F1={cand_metrics['f1']} / Acc={cand_metrics['accuracy']} "
                        f"did not beat current F1={curr_metrics['f1']} / Acc={curr_metrics['accuracy']}"
                    ),
                )
                deployed = False
                decision_msg = (
                    f"Candidate model {model_version} rejected (did not meet performance criteria). "
                    "Existing production model retained."
                )

            logger.info(decision_msg)
            return {
                "success": True,
                "deployed": deployed,
                "model_version": model_version,
                "dataset_version": ds_version,
                "candidate_metrics": cand_metrics,
                "current_metrics": curr_metrics,
                "message": decision_msg,
            }
        except Exception as exc:
            logger.exception("Model retraining failed")
            return {"success": False, "error": str(exc)}
        finally:
            self._lock.release()

    def rollback_to_version(self, target_version: str) -> dict:
        """Roll back production model and matching vectorizer to a previous version."""
        with self._lock:
            ver_info = self.storage.get_model_version(target_version)
            if not ver_info:
                return {"success": False, "error": f"Model version '{target_version}' not found."}

            m_path = ver_info["model_path"]
            v_path = ver_info["vectorizer_path"]
            if not os.path.exists(m_path) or not os.path.exists(v_path):
                return {"success": False, "error": f"Artifacts for version '{target_version}' are missing on disk."}

            shutil.copy2(m_path, self.storage.prod_model_path)
            shutil.copy2(v_path, self.storage.prod_vectorizer_path)
            self.storage.set_active_model_version(target_version)

            if self.on_model_updated:
                self.on_model_updated(self.storage.prod_model_path, self.storage.prod_vectorizer_path)

            logger.info("Rolled back production model to version %s", target_version)
            return {
                "success": True,
                "active_version": target_version,
                "message": f"Successfully rolled back to model {target_version}.",
            }
