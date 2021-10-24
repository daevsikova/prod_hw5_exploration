import argparse


class BaseOptions():
    def __init__(self, cmd_line=None):
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
        
        # model parameters
        parser.add_argument("-c", "--config", required=True, type=str, help="Path to config file to overwrite argparse params")
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return self.opt
