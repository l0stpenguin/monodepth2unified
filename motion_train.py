from __future__ import absolute_import, division, print_function

from motion_trainer import MotionTrainer
from motion_options import MotionOptions
import setproctitle

options = MotionOptions()
opts = options.parse()
setproctitle.setproctitle(opts.model_name)

if __name__ == "__main__":
    trainer = MotionTrainer(opts)
    trainer.train()