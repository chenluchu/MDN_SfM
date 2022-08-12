# Code based on Monodepth2

from __future__ import absolute_import, division, print_function
from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    # Train model
    trainer = Trainer(opts)
    trainer.train()

    # # choose hyper-parameter in model
    # trainer = Trainer(opts)
    # trainer.hyperparameter_try("w_d2_sim")

    # # compute epipolar map normalization threshold over entire training set
    # thresholds = trainer.epipolar_statics()
    # print("Thresholds are :", thresholds)
