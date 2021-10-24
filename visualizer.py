import os
import sys
import tempfile
import time
import uuid
from datetime import datetime

import imageio
import numpy as np
import wandb

from . import util

VisdomExceptionBase = Exception if sys.version_info[0] == 2 else ConnectionError


if not os.environ.get("WANDB_API_KEY", None):
    os.environ["WANDB_API_KEY"] = "e891f26c3ad7fd5a7e215dc4e344acc89c8861da"


class Visualizer():
    """
    This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_wandb = opt.isTrain and opt.use_wandb
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False

        if self.use_wandb:
            name = opt.name + datetime.strftime(datetime.now(), "_%h%d_%H%M%S")
            project = "prod_hw5_rl"
            wandb.init(project=project, entity="daevsikova", config=opt, name=name)

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, step, prefix="train"):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """

        if self.use_wandb:
            with tempfile.TemporaryDirectory(dir=".") as tmp:
                len_ = len(visuals)
                for i, (label, image_tens) in enumerate(visuals.items()):
                    label = prefix + "_" + label
                    fname = f"{tmp}/{str(uuid.uuid4())[:8]}_{label}.jpg"
                    imageio.imsave(fname, util.tensor2im(image_tens))
                    wandb.log({label: wandb.Image(fname)}, step=step, commit=(i == len_ - 1))

    def plot_current_losses(self, epoch, counter_ratio, losses, step, prefix=None):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if len(losses) == 0:
            return

        if self.use_wandb:
            for tag, value in losses.items():
                if prefix is not None:
                    tag = prefix + tag

                wandb.log({tag: value}, step=step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
