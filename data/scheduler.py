"""
Simple daily pipeline runner.
Run once: python data/scheduler.py
It will re-run the pipeline every 24 hours.
For production: replace with cron or APScheduler.
"""

import schedule
import time
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.kaggle_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def job():
    log.info("Scheduled pipeline run starting...")
    run_pipeline(load_db=False)
    log.info("Scheduled run complete.")


# Run immediately once, then every 24 hours
job()
schedule.every(24).hours.do(job)

log.info("Scheduler running. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)