import logging
import logstash
from datetime import datetime

def make_logger(name=None):
    logger = logging.getLogger(name)
 
    logger.setLevel(logging.DEBUG)
 
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
     
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename="logs/{:%Y-%m-%d}.log".format(datetime.now()))
     
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
 
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
 
    logger.addHandler(console)
    logger.addHandler(file_handler)
 
    stash = logstash.TCPLogstashHandler('localhost',5000,version=1)
    stash.setFormatter(formatter)
    logger.addHandler(stash)
 
    return logger