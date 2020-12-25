# -*- coding:utf-8 -*-
import torch
import random
import iter_utils
from torch.utils.data.dataset import Dataset


class ASDataset(Dataset):
    """
        Dataset for MST-AISHELL-3
    """

    def __init__(self, params, mode="train"):
        self.dataset_path = params.dataset_path
        self.set_mode(mode)
        self.data_list = iter_utils.load_text(
            self.dataset_path + "/content.txt")
        random.seed(params.random_seed)
        random.shuffle(self.data_list)
        self.data_list = self.data_list[:params.num_data]
        iter_utils.print_INFO(
            "dataset", "load {} data list successfully".format(self.mode))

    def set_mode(self, mode):
        assert mode in ("train", "val", "test")
        self.mode = mode
        if mode == "train" or mode == "val":
            self.dataset_path += "/train"
        elif mode == "test":
            self.dataset_path += "/test"

    def __getitem__(self, index):
        name, text = self.data_list[index].split("\t")
        text = text[:-1]
        path = self.dataset_path + "/wav/{}/{}".format(name[:7], name)
        return (text, path)

    def __len__(self):
        return len(self.data_list)
