import logging
import multiprocessing as mp
from logging.handlers import QueueListener
from multiprocessing.context import BaseContext

_log_queue: mp.Queue[logging.LogRecord] | None = None
_log_listener = None


class InjectHandler(logging.Handler):
    "Handler that re-injects a message into parent process logging"

    level = logging.DEBUG

    def handle(self, record: logging.LogRecord) -> bool:
        logger = logging.getLogger(record.name)
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)
            return True
        else:
            return False


def log_queue(ctx: BaseContext) -> mp.Queue[logging.LogRecord]:
    """
    Get the log queue for child process logging.
    """
    global _log_queue, _log_listener

    if _log_queue is None:
        _log_queue = ctx.Queue()
        _log_listener = QueueListener(_log_queue, InjectHandler())
        _log_listener.start()
    return _log_queue
