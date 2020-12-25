# -*- coding:utf-8 -*-
import os
import sys
import torch
import librosa


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def load_wav(path, sr):
    return librosa.load(path, sr=sr)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.readlines()
    return text


def print_ERROR(location, message):
    print("\033[1;31m[ERROR]\033[0m\033[1;36m[{}]\033[0m: {}".format(
        location, message))


def print_WARNING(location, message):
    print("\033[1;33m[WARNING]\033[0m\033[1;36m[{}]\033[0m: {}".format(
        location, message))


def print_INFO(location, message):
    print("\033[1;34m[INFO]\033[0m\033[1;36m[{}]\033[0m: {}".format(
        location, message))


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


class Logger():
    def __init__(self, file_path, mode="a", output_terminal=True):
        self.terminal = sys.stdout
        self.output_terminal = output_terminal
        try:
            self.file = open(file_path, mode, encoding="utf-8")
        except:
            print_ERROR("logger", "file does not exist")
            raise
        else:
            print_INFO("logger", "logger starts")

    def write(self, message):
        cur_time = time.strftime("%H:%M:%S", time.localtime())
        message = "{} {}\n".format(cur_time, message)
        if self.output_terminal:
            self.terminal.write(message)
        self.file.write(message)

    def clear(self):
        self.file.truncate(0)

    def close(self):
        self.file.close()
        print_INFO("logger", "logger closes")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
