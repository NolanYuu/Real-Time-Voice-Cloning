# -*- coding:utf-8 -*-
from easydict import EasyDict as edict


def get_params(use_json=False, setting_str=None):
    params = edict(
        # ------------------------------
        # overall parameters
        dataset_path="/inspur/AISHELL-3/",
        log_path="/inspur/Real-Time-Voice-Cloning/my_log.txt",

        learning_rate=1e-4,
        random_seed=1,
        num_data=500,
        batch_size=32,
    )

    return params
