import atexit

from apscheduler.schedulers.background import BackgroundScheduler

from .retraining import run_retraining


def start_retraining_schedule():
	scheduler = BackgroundScheduler()
	scheduler.add_job(run_retraining, 'cron', hour=8, minute=0)
	scheduler.start()
	atexit.register(lambda: scheduler.shutdown())
