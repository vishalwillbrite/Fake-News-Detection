"""
scheduler_service.py
APScheduler background scheduler for periodic news collection and
automatic retraining checks.
"""

import logging
import os
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from services.model_retraining_service import ModelRetrainingService
from services.news_collection_service import NewsCollectionService

logger = logging.getLogger(__name__)


class ContinuousLearningScheduler:
    """Wraps APScheduler BackgroundScheduler with clean start/stop controls."""

    def __init__(
        self,
        collection_service: NewsCollectionService,
        retraining_service: ModelRetrainingService,
        interval_minutes: int = 60,
    ):
        self.collection_service = collection_service
        self.retraining_service = retraining_service
        self.interval_minutes = interval_minutes
        self._scheduler: Optional[BackgroundScheduler] = None

    def _run_scheduled_cycle(self):
        logger.info("Running scheduled news collection cycle...")
        self.collection_service.collect_news(trigger_type="scheduled")
        # Retraining only runs if min_approved_samples threshold is met
        self.retraining_service.retrain_if_ready()

    def start(self):
        if os.environ.get("VERITAS_DISABLE_SCHEDULER") == "1":
            logger.info("Scheduler disabled via VERITAS_DISABLE_SCHEDULER=1")
            return
        if self._scheduler and self._scheduler.running:
            return

        self._scheduler = BackgroundScheduler(daemon=True)
        self._scheduler.add_job(
            self._run_scheduled_cycle,
            trigger="interval",
            minutes=self.interval_minutes,
            id="veritas_news_collection_job",
            replace_existing=True,
            max_instances=1,
        )
        self._scheduler.start()
        logger.info(
            "ContinuousLearningScheduler started (interval=%d minutes)",
            self.interval_minutes,
        )

    def stop(self):
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("ContinuousLearningScheduler stopped.")

    def is_running(self) -> bool:
        return bool(self._scheduler and self._scheduler.running)
