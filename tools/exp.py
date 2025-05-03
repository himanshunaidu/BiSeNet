import logging
import sys

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info('Start training')
logger.error('Error message')