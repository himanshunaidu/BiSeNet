import sys
sys.path.insert(0, '.')
import logging

from configs import set_cfg_from_file

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info('Start training')
logger.error('Error message')

cfg = set_cfg_from_file("configs/bisenetv1_city.py")
print(cfg.__dict__)