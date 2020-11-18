import datetime, pytz
from dateutil.tz import tzlocal

log_dir = None
verbose = False

def log(message):
    ts = pytz.utc.localize(datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    if verbose:
        print(f'{ts} {message}')
    if log_dir is not None:
        print(f'{ts} {message}', file=open(log_dir, 'a'))
