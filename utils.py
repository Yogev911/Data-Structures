import datetime
from functools import wraps
import logzero
from logzero import logger

# Setup rotating logfile with 3 rotations, each with a maximum filesize of 1MB:
logzero.logfile("logfile.log", maxBytes=2048, backupCount=3)

# Log messages


def timer(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        start = datetime.datetime.now()
        r = f(*args, **kwargs)
        print((datetime.datetime.now() - start).microseconds)
        return r

    return wrapped

if __name__ == '__main__':
    logger.info("This log message goes to the console and the logfile")