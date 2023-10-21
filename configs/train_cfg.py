import os
from easydict import EasyDict
from configs.kmnist_cfg import cfg as dataset_cfg
from configs.mlp_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.batch_size = 16

cfg.model_cfg = model_cfg
cfg.dataset_cfg = dataset_cfg

cfg.exp_dir = os.path.join(ROOT_DIR, 'mlp_model')
