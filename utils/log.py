import logging
import logstash
from datetime import datetime

def make_logger(name=None):
    logger = logging.getLogger(name, )
 
    logger.setLevel(logging.INFO)
 
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
     
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    
    # file_handler = logging.FileHandler(filename="logs/{:%Y-%m-%d}.log".format(datetime.now()))
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    
    timedfilehandler = logging.handlers.TimedRotatingFileHandler(filename='logs/user_id_recsys.log'.format(datetime.now()), when='midnight', interval=1, encoding='utf-8')
    timedfilehandler.setFormatter(formatter)   
    timedfilehandler.suffix = "%Y-%m-%d"
    
    # logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.addHandler(timedfilehandler)
 
    # stash = logstash.TCPLogstashHandler('localhost',5000,version=1)
    # stash.setFormatter(formatter)
    # logger.addHandler(stash)
 
    return logger