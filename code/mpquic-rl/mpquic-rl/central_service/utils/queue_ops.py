# Simple queue operations
# get or put a request to queue
# blocking operation with a small timeout
import queue
from threading import Event
import multiprocessing as mp

import queue
import multiprocessing as mp
import logging

def get_request(q: queue.Queue, logger: logging.Logger, end_of_run: mp.Event = None):
    logger.info("Waiting for request...")
    while not end_of_run.is_set():
        try:
            req, evt = q.get(timeout=0.05)
            logger.info(f"Received request: {req}")
            return req, evt
        except queue.Empty:
            continue
    logger.info("End of run detected in get_request.")
    return None, None


def put_response(response, queue: queue.Queue, logger):
    logger.info("Putting response...")
    try:
        queue.put(response)
    except Exception as ex:
        logger.error(ex)