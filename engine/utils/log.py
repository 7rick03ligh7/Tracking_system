import os
from functools import wraps
from typing import Any, Callable
import logging
from logging.handlers import RotatingFileHandler

LOG_LEVEL = logging.DEBUG
LOG_FORMAT = "[%(asctime)-23s] %(levelname)-9s [%(name)s." \
    "{%(filename)s}.%(funcName)s:%(lineno)d] %(message)s"
LOG_NAME = "$default.log"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt="%H:%M:%S")
LOGGER = logging.getLogger(LOG_NAME)
FILE_HANDLER = RotatingFileHandler(
    os.path.join(os.path.dirname(__file__), LOG_NAME),
    maxBytes=1024 * 100,
    backupCount=10
)
FILE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))
FILE_HANDLER.setLevel(LOG_LEVEL)
LOGGER.addHandler(FILE_HANDLER)
LOGGER.debug('=' * 100)


# def log(func: Callable) -> Callable:
#     """ Logging function/method behavior """

#     @wraps(func)
#     def wrapper(*args, **kwargs) -> Any:
#         LOGGER.debug("<%s> ENTER ---->", func.__name__)
#         # for arg in args:
#         #     LOGGER.debug("<%s> ARG  \t%s", func.__name__, arg)
#         # for key, value in kwargs.items():
#         #     LOGGER.debug("<%s> KWARG\t%s=%s", func.__name__, key, value)
#         result = func(*args, **kwargs)
#         LOGGER.debug("<%s> EXIT ----x\t%s", func.__name__, result)
#         return result
#     return wrapper


def log(func: Callable) -> Callable:
    """ Logging function/method behavior """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        LOGGER.debug("<%s> ENTER ---->", func.__name__)
        result = func(*args, **kwargs)
        LOGGER.debug("<%s> EXIT -----x", func.__name__)
        return result
    return wrapper
