import os, sys, time
import hashlib
import coadapt
import experiment_configs as cfg
import json
import random
import numpy as np
import torch

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config):

    # set seed
    seed_torch(config['seed'])
    co = coadapt.Coadaptation(config)
    co.run()



if __name__ == "__main__":
    config = cfg.config_dict['td3_bo_batch']
    # config = cfg.config_dict['td3_bo_sim']

    main(config)
