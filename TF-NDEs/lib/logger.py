import logging
import time
import os
from datetime import datetime, timedelta, timezone

# === 自定义北京时间 ===
BEIJING_TZ = timezone(timedelta(hours=0))
def beijing_time(*args):
    return datetime.now(BEIJING_TZ).timetuple()

def get_logger(log_dir, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 强制使用北京时间
    logging.Formatter.converter = beijing_time
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M:%S")


    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)

    if not debug:
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, 'run.log')
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger