import os
from easydict import EasyDict
from configs.kmnist_cfg import cfg as dataset_cfg
from configs.mlp_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.batch_size = 64
cfg.lr = 1e-2
cfg.optimizer_name = 'SGD'  # ['SGD', 'Adam']

cfg.model_cfg = model_cfg
cfg.dataset_cfg = dataset_cfg

cfg.load_existing_model = True
cfg.exp_dir = os.path.join(ROOT_DIR, 'mlp_model')

cfg.epochs = 10

cfg.last_experiment = 0
