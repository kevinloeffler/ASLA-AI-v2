import atexit

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .retraining import run_retraining


def start_retraining_schedule():
	scheduler = BackgroundScheduler()
	scheduler.add_job(run_retraining, CronTrigger(day_of_week='sat', hour=2, minute=0))
	scheduler.start()
	atexit.register(lambda: scheduler.shutdown())
