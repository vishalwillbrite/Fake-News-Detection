"""
pipeline_storage.py
Thread-safe SQLite and versioned dataset/model storage for VERITAS.AI
continuous learning pipeline. Preserves original datasets untouched and
maintains a fixed, untouched holdout test set to prevent data leakage.
"""

from contextlib import contextmanager
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_text

logger = logging.getLogger(__name__)

VALID_STATUSES = {"PENDING", "APPROVED", "REJECTED", "NEEDS_REVIEW"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_url(url: str) -> str:
    norm = (url or "").strip().split("#")[0].rstrip("/").lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def hash_content(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class PipelineStorage:
    """Manages SQLite state, dataset snapshots, holdout test set, and model versions."""

    def __init__(
        self,
        base_dir: str = "data_pipeline",
        fake_csv: str = "dataset/Fake.csv",
        true_csv: str = "dataset/True.csv",
        prod_model_path: str = "models/fake_news_model.pkl",
        prod_vectorizer_path: str = "models/tfidf_vectorizer.pkl",
        model_versions_dir: str = "models/versions",
    ):
        self.base_dir = base_dir
        self.db_path = os.path.join(base_dir, "pipeline.db")
        self.datasets_dir = os.path.join(base_dir, "datasets")
        self.train_base_csv = os.path.join(self.datasets_dir, "train_base.csv")
        self.holdout_csv = os.path.join(self.datasets_dir, "holdout_test.csv")

        self.fake_csv = fake_csv
        self.true_csv = true_csv
        self.prod_model_path = prod_model_path
        self.prod_vectorizer_path = prod_vectorizer_path
        self.model_versions_dir = model_versions_dir

        self._lock = threading.RLock()
        self._original_hashes: Set[str] = set()
        self._holdout_hashes: Set[str] = set()
        self._original_counts: Dict[str, int] = {"total": 0, "real": 0, "fake": 0}

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.model_versions_dir, exist_ok=True)

        self._init_db()
        self._init_base_datasets_and_baseline_model()

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    url_hash TEXT UNIQUE NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    publisher TEXT,
                    domain TEXT,
                    published_date TEXT,
                    collected_at TEXT NOT NULL,
                    proposed_label INTEGER,
                    status TEXT NOT NULL,
                    label_source TEXT NOT NULL,
                    verification_evidence TEXT,
                    source_url TEXT,
                    dataset_version TEXT,
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    review_note TEXT
                );

                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    original_count INTEGER NOT NULL,
                    approved_added_count INTEGER NOT NULL,
                    total_count INTEGER NOT NULL,
                    real_count INTEGER NOT NULL,
                    fake_count INTEGER NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    vectorizer_path TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    confusion_matrix_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS collection_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    trigger_type TEXT NOT NULL,
                    queries_run INTEGER NOT NULL DEFAULT 0,
                    fetched_count INTEGER NOT NULL DEFAULT 0,
                    duplicates_skipped INTEGER NOT NULL DEFAULT 0,
                    new_candidates_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT
                );
                """
            )

    def _init_base_datasets_and_baseline_model(self):
        """
        Read original Fake.csv / True.csv (read-only), create fixed untouched
        holdout_test.csv if missing, and register baseline v1.0 model.
        """
        if os.path.exists(self.fake_csv) and os.path.exists(self.true_csv):
            fake_df = pd.read_csv(self.fake_csv)
            true_df = pd.read_csv(self.true_csv)
            fake_df["label"] = 0
            true_df["label"] = 1

            self._original_counts = {
                "total": int(len(fake_df) + len(true_df)),
                "fake": int(len(fake_df)),
                "real": int(len(true_df)),
            }

            combined = pd.concat([fake_df, true_df], ignore_index=True)
            combined["text"] = combined["text"].fillna("").astype(str)
            combined["title"] = combined.get("title", pd.Series([""] * len(combined))).fillna("").astype(str)
            combined["content_hash"] = combined["text"].apply(hash_content)
            combined = combined.drop_duplicates(subset=["content_hash"]).reset_index(drop=True)
            self._original_hashes = set(combined["content_hash"].tolist())

            if not os.path.exists(self.holdout_csv) or not os.path.exists(self.train_base_csv):
                train_df, holdout_df = train_test_split(
                    combined[["title", "text", "label", "content_hash"]],
                    test_size=0.2,
                    random_state=42,
                    stratify=combined["label"],
                )
                train_df.to_csv(self.train_base_csv, index=False)
                holdout_df.to_csv(self.holdout_csv, index=False)
                logger.info(
                    "Initialized base split: %d train_base, %d fixed holdout_test",
                    len(train_df),
                    len(holdout_df),
                )

            holdout_loaded = pd.read_csv(self.holdout_csv)
            if "content_hash" in holdout_loaded.columns:
                self._holdout_hashes = set(holdout_loaded["content_hash"].astype(str).tolist())
            else:
                self._holdout_hashes = {hash_content(t) for t in holdout_loaded["text"].fillna("").astype(str)}

        # Record baseline v1 dataset & v1.0 model if empty
        with self._conn() as conn:
            dv_count = conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0]
            if dv_count == 0:
                conn.execute(
                    """
                    INSERT INTO dataset_versions
                    (version, created_at, original_count, approved_added_count, total_count, real_count, fake_count, snapshot_path, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "v1",
                        utc_now_iso(),
                        self._original_counts["total"],
                        0,
                        self._original_counts["total"],
                        self._original_counts["real"],
                        self._original_counts["fake"],
                        self.train_base_csv,
                        "Original baseline dataset (read-only reference)",
                    ),
                )

            mv_count = conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
            if mv_count == 0 and os.path.exists(self.prod_model_path) and os.path.exists(self.prod_vectorizer_path):
                v1_dir = os.path.join(self.model_versions_dir, "v1.0")
                os.makedirs(v1_dir, exist_ok=True)
                v1_model = os.path.join(v1_dir, "fake_news_model.pkl")
                v1_vec = os.path.join(v1_dir, "tfidf_vectorizer.pkl")
                shutil.copy2(self.prod_model_path, v1_model)
                shutil.copy2(self.prod_vectorizer_path, v1_vec)

                metrics = self.evaluate_artifacts_on_holdout(v1_model, v1_vec)
                conn.execute(
                    """
                    INSERT INTO model_versions
                    (version, created_at, dataset_version, model_path, vectorizer_path, accuracy,
                     precision_score, recall_score, f1_score, confusion_matrix_json, is_active, status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'DEPLOYED', ?)
                    """,
                    (
                        "v1.0",
                        utc_now_iso(),
                        "v1",
                        v1_model,
                        v1_vec,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        json.dumps(metrics["confusion_matrix"]),
                        "Initial baseline production Logistic Regression model",
                    ),
                )

    def evaluate_artifacts_on_holdout(self, model_path: str, vectorizer_path: str) -> dict:
        """Evaluate a (model, vectorizer) pair on the fixed untouched holdout_test.csv."""
        if not os.path.exists(self.holdout_csv):
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]]}

        holdout_df = pd.read_csv(self.holdout_csv)
        cleaned = holdout_df["text"].fillna("").astype(str).apply(preprocess_text)
        y_true = holdout_df["label"].astype(int).tolist()

        model = joblib.load(model_path)
        vec = joblib.load(vectorizer_path)
        X = vec.transform(cleaned)
        y_pred = model.predict(X)

        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def is_duplicate(self, url: str, content: str) -> bool:
        uh = hash_url(url)
        ch = hash_content(content)
        if ch in self._original_hashes or ch in self._holdout_hashes:
            return True
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM candidates WHERE url_hash = ? OR content_hash = ? LIMIT 1",
                (uh, ch),
            ).fetchone()
            return row is not None

    def is_in_holdout(self, content: str) -> bool:
        return hash_content(content) in self._holdout_hashes

    def add_candidate(
        self,
        url: str,
        title: str,
        content: str,
        publisher: str,
        domain: str,
        published_date: Optional[str],
        proposed_label: Optional[int],
        status: str,
        label_source: str,
        verification_evidence: dict,
        source_url: Optional[str] = None,
    ) -> Optional[int]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        if self.is_duplicate(url, content):
            return None

        uh = hash_url(url)
        ch = hash_content(content)
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO candidates
                (url, url_hash, content_hash, title, content, publisher, domain, published_date,
                 collected_at, proposed_label, status, label_source, verification_evidence, source_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url,
                    uh,
                    ch,
                    title,
                    content,
                    publisher,
                    domain,
                    published_date,
                    utc_now_iso(),
                    proposed_label,
                    status,
                    label_source,
                    json.dumps(verification_evidence),
                    source_url or url,
                ),
            )
            return cur.lastrowid

    def list_candidates(self, status: Optional[str] = None, limit: int = 100) -> List[dict]:
        with self._conn() as conn:
            if status and status in VALID_STATUSES:
                rows = conn.execute(
                    "SELECT * FROM candidates WHERE status = ? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM candidates ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        items = []
        for r in rows:
            d = dict(r)
            try:
                d["verification_evidence"] = json.loads(d["verification_evidence"] or "{}")
            except Exception:
                d["verification_evidence"] = {}
            items.append(d)
        return items

    def update_candidate_status(
        self,
        candidate_id: int,
        status: str,
        label: Optional[int] = None,
        reviewed_by: str = "admin",
        review_note: str = "",
    ) -> bool:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        if status == "APPROVED" and label not in (0, 1):
            raise ValueError("Approved candidates must have label 0 (Fake) or 1 (Real).")

        with self._conn() as conn:
            existing = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
            if not existing:
                return False

            final_label = label if label in (0, 1) else existing["proposed_label"]
            if status == "APPROVED" and final_label not in (0, 1):
                raise ValueError("Cannot approve candidate without a valid label (0 or 1).")

            conn.execute(
                """
                UPDATE candidates
                SET status = ?, proposed_label = ?, reviewed_by = ?, reviewed_at = ?, review_note = ?,
                    label_source = CASE WHEN ? != '' THEN ? ELSE label_source END
                WHERE id = ?
                """,
                (
                    status,
                    final_label,
                    reviewed_by,
                    utc_now_iso(),
                    review_note,
                    "human_review" if reviewed_by else "",
                    "human_review",
                    candidate_id,
                ),
            )
            return True

    def get_approved_articles(self) -> List[dict]:
        """Return all APPROVED candidates with valid labels (0/1), excluding holdout hashes."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE status = 'APPROVED' AND proposed_label IN (0, 1) ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows if r["content_hash"] not in self._holdout_hashes]

    def count_untrained_approved(self) -> int:
        with self._conn() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM candidates WHERE status = 'APPROVED' AND proposed_label IN (0, 1) AND dataset_version IS NULL"
                ).fetchone()[0]
            )

    def mark_approved_used_in_version(self, candidate_ids: List[int], dataset_version: str):
        if not candidate_ids:
            return
        placeholders = ",".join("?" for _ in candidate_ids)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE candidates SET dataset_version = ? WHERE id IN ({placeholders}) AND dataset_version IS NULL",
                [dataset_version, *candidate_ids],
            )

    def next_dataset_version(self) -> str:
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0]
        return f"v{count + 1}"

    def next_model_version(self) -> str:
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
        return f"v1.{count}"

    def record_dataset_version(
        self,
        version: str,
        original_count: int,
        approved_added_count: int,
        total_count: int,
        real_count: int,
        fake_count: int,
        snapshot_path: str,
        notes: str = "",
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO dataset_versions
                (version, created_at, original_count, approved_added_count, total_count, real_count, fake_count, snapshot_path, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (version, utc_now_iso(), original_count, approved_added_count, total_count, real_count, fake_count, snapshot_path, notes),
            )

    def record_model_version(
        self,
        version: str,
        dataset_version: str,
        model_path: str,
        vectorizer_path: str,
        metrics: dict,
        is_active: bool,
        status: str,
        notes: str = "",
    ):
        with self._conn() as conn:
            if is_active:
                conn.execute("UPDATE model_versions SET is_active = 0")
            conn.execute(
                """
                INSERT INTO model_versions
                (version, created_at, dataset_version, model_path, vectorizer_path, accuracy,
                 precision_score, recall_score, f1_score, confusion_matrix_json, is_active, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version,
                    utc_now_iso(),
                    dataset_version,
                    model_path,
                    vectorizer_path,
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    json.dumps(metrics["confusion_matrix"]),
                    1 if is_active else 0,
                    status,
                    notes,
                ),
            )

    def get_active_model_version(self) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(row)
        d["confusion_matrix"] = json.loads(d.get("confusion_matrix_json") or "[[0,0],[0,0]]")
        return d

    def get_model_version(self, version: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM model_versions WHERE version = ? LIMIT 1", (version,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["confusion_matrix"] = json.loads(d.get("confusion_matrix_json") or "[[0,0],[0,0]]")
        return d

    def set_active_model_version(self, version: str):
        with self._conn() as conn:
            conn.execute("UPDATE model_versions SET is_active = 0")
            conn.execute("UPDATE model_versions SET is_active = 1, status = 'DEPLOYED' WHERE version = ?", (version,))

    def record_collection_run(
        self,
        started_at: str,
        trigger_type: str,
        queries_run: int,
        fetched_count: int,
        duplicates_skipped: int,
        new_candidates_count: int,
        status: str,
        error_message: str = "",
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO collection_runs
                (started_at, finished_at, trigger_type, queries_run, fetched_count,
                 duplicates_skipped, new_candidates_count, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    utc_now_iso(),
                    trigger_type,
                    queries_run,
                    fetched_count,
                    duplicates_skipped,
                    new_candidates_count,
                    status,
                    error_message,
                ),
            )

    def get_dashboard_stats(self) -> dict:
        with self._conn() as conn:
            status_rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM candidates GROUP BY status"
            ).fetchall()
            status_counts = {r["status"]: r["cnt"] for r in status_rows}

            approved_dist = conn.execute(
                "SELECT proposed_label, COUNT(*) as cnt FROM candidates WHERE status = 'APPROVED' GROUP BY proposed_label"
            ).fetchall()
            app_dist_map = {r["proposed_label"]: r["cnt"] for r in approved_dist}

            last_run = conn.execute(
                "SELECT * FROM collection_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()

            ds_versions = [
                dict(r) for r in conn.execute("SELECT * FROM dataset_versions ORDER BY id DESC").fetchall()
            ]
            m_versions = []
            for r in conn.execute("SELECT * FROM model_versions ORDER BY id DESC").fetchall():
                d = dict(r)
                d["confusion_matrix"] = json.loads(d.get("confusion_matrix_json") or "[[0,0],[0,0]]")
                m_versions.append(d)

        total_collected = sum(status_counts.values())
        approved_real = app_dist_map.get(1, 0)
        approved_fake = app_dist_map.get(0, 0)

        return {
            "original_articles": self._original_counts["total"],
            "original_real": self._original_counts["real"],
            "original_fake": self._original_counts["fake"],
            "newly_collected": total_collected,
            "approved_articles": status_counts.get("APPROVED", 0),
            "pending_articles": status_counts.get("PENDING", 0),
            "needs_review_articles": status_counts.get("NEEDS_REVIEW", 0),
            "rejected_articles": status_counts.get("REJECTED", 0),
            "untrained_approved": self.count_untrained_approved(),
            "total_real_distribution": self._original_counts["real"] + approved_real,
            "total_fake_distribution": self._original_counts["fake"] + approved_fake,
            "last_collection": dict(last_run) if last_run else None,
            "dataset_versions": ds_versions,
            "model_versions": m_versions,
            "active_model": self.get_active_model_version(),
        }
