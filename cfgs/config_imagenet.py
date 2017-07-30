import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.exemplar_size = 127
cfg.search_size = 255

cfg.score_size = 17

cfg.pos_radius = 2
cfg.label_weight_type = 'balanced'

# cfg.steps_per_epoch = 5e4
cfg.steps_per_epoch = 100
cfg.max_epoch = 50

cfg.weight_decay = 5e-4
