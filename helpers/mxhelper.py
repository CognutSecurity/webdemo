# -*- coding=utf-8 -*-


import argparse


def _device_list(s):
    l = s.split(',')
    devices = []
    for i in l:
        try:
            if int(i) < 0:
                msg = 'Invalid Device number, try input e.g., --gpus 0,2,3'
                raise argparse.ArgumentTypeError(msg)
            else:
                devices.append(int(i))
        except ValueError:
            msg = 'Invalid Device number, try input e.g., --gpus 0,2,3'
            raise argparse.ArgumentTypeError(msg)
    return devices


class BasicArgparser(object):

    def __init__(self, prog):

        argparser = argparse.ArgumentParser(prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        argparser.add_argument('--epochs', '-e', type=int,
                               default=10, help='Number of epochs to run')
        argparser.add_argument('--batch_size', '-bs', type=int,
                               default=64, help='Batch size ')
        argparser.add_argument('--gpus', type=_device_list,
                               help='IDs of GPUs to be used')
        argparser.add_argument('--cpus', type=_device_list,
                               help='IDs of CPUs to be used')
        argparser.add_argument('--learning_rate', '-lr', type=float, default=.01,
                               help='Learning rate defautls to 0.01')
        self.parser = argparser

    def get_parser(self):

        return self.parser
