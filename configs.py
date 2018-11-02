"""Global configs."""
import logging
from chino.configurator import cfg

logger = logging.getLogger(__name__)

cfg.OUTPUT_PATH = '.default'

cfg.DATA_PATH = '.data'

cfg.DATASET = 'modelnet40'

cfg.NUM_POINTS = 1024

cfg.EPOCHS = 250

cfg.RNG_SEED = 9527

cfg.NUM_WORKERS = 4

cfg.BATCH_SIZE = 32

cfg.LR = 1e-3

cfg.MIN_LR = 1e-5

cfg.MOMENTUM = 0.9

cfg.WEIGHT_DECAY = 5e-4

cfg.GAMMA = 0.7

cfg.STEPSIZE = 20

cfg.OPTIMIZER = 'adam'

cfg.SNAPSHOT = -1

cfg.TEST_INTERVAL = 1

cfg.TEST_BATCH_SIZE = 4

cfg.BN_MOMENTUM = 0.5

cfg.BN_STEPSIZE = 20

cfg.BN_GAMMA = 0.5

cfg.BN_MIN_MOMENTUM = 0.01

cfg.freeze()
